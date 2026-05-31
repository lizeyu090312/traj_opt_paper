import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


LATENT_SCALE = 0.18215


def is_packed_vae_latent_dir(path):
    return (
        os.path.isfile(os.path.join(path, 'latents.bin'))
        and os.path.isfile(os.path.join(path, 'meta.npz'))
    )


def sample_packed_vae_latents(mean_std, latent_scale=LATENT_SCALE):
    if mean_std.ndim != 4 or mean_std.shape[1] != 8:
        raise ValueError(
            f"Expected packed VAE latents with shape [N, 8, H, W], got {tuple(mean_std.shape)}"
        )
    mean, std = mean_std.chunk(2, dim=1)
    return (mean + torch.randn_like(mean) * std) * latent_scale


def paired_packed_latent_original_count(num_records):
    if num_records % 2 != 0:
        raise ValueError(
            "One-view-per-image packed-latent sampling expects an even number of stored views "
            f"(got {num_records})."
        )
    return num_records // 2


def resolve_packed_latent_view_mode(requested_mode, num_records, *, expected_original_count=None, tolerance=0):
    valid_modes = {"auto", "all", "one-per-image"}
    if requested_mode not in valid_modes:
        raise ValueError(
            f"Unsupported packed-latent view mode {requested_mode!r}; expected one of {sorted(valid_modes)}."
        )
    if requested_mode == "all":
        return "all"
    if requested_mode == "one-per-image":
        paired_packed_latent_original_count(num_records)
        return "one-per-image"
    if expected_original_count is None or num_records % 2 != 0:
        return "all"
    if num_records >= max(0, (2 * int(expected_original_count)) - int(tolerance)):
        return "one-per-image"
    return "all"


class PairedPackedLatentSampler(Sampler):
    """
    Sampler for packed latent datasets that store two pre-encoded views per source
    image in consecutive slots: index 2*i is the original view and 2*i+1 is the
    horizontally flipped view produced during preprocessing.

    Each epoch, this sampler can either choose one random stored view per source
    image (`view_policy="random"`) or always use the canonical original view
    (`view_policy="first"`). The resulting sample order is deterministic under
    `set_epoch(epoch)`, so resume and DDP fast-forward stay aligned.
    """

    def __init__(self, *, num_originals=None, original_indices=None, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=False, view_policy="random"):
        if (num_originals is None) == (original_indices is None):
            raise ValueError("Provide exactly one of num_originals or original_indices.")
        if num_replicas < 1:
            raise ValueError("num_replicas must be at least 1.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas={num_replicas}.")
        if view_policy not in {"random", "first"}:
            raise ValueError(f"Unsupported view_policy: {view_policy}")

        if original_indices is not None:
            self._original_indices = tuple(int(idx) for idx in original_indices)
            if len(self._original_indices) < 1:
                raise ValueError("original_indices must not be empty.")
            self._num_originals = len(self._original_indices)
        else:
            self._original_indices = None
            self._num_originals = int(num_originals)
            if self._num_originals < 1:
                raise ValueError("num_originals must be at least 1.")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.view_policy = view_policy
        self.epoch = 0

        if self.drop_last and self._num_originals % self.num_replicas != 0:
            self.num_samples = math.ceil((self._num_originals - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self._num_originals / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _ordered_original_indices(self, generator):
        if self.shuffle:
            order = torch.randperm(self._num_originals, generator=generator).tolist()
        else:
            order = list(range(self._num_originals))
        if self._original_indices is None:
            return order
        return [self._original_indices[idx] for idx in order]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        original_indices = self._ordered_original_indices(generator)
        if self.view_policy == "random":
            view_offsets = torch.randint(0, 2, (len(original_indices),), generator=generator).tolist()
            indices = [2 * original_idx + view_offset for original_idx, view_offset in zip(original_indices, view_offsets)]
        else:
            indices = [2 * original_idx for original_idx in original_indices]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    repeats = math.ceil(padding_size / len(indices))
                    indices += (indices * repeats)[:padding_size]
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Sampler produced {len(indices)} samples for rank {self.rank}, expected {self.num_samples}."
            )
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class ImgVAELabelDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = os.path.join(data_dir, 'images')
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])


class VAELabelDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self.features_dir = os.path.join(data_dir, 'vae-sd')

        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        feature_fname = self.feature_fnames[idx]
        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(features), torch.tensor(self.labels[idx])


class PackedVAELabelDataset(Dataset):
    # Memmap-backed VAE latent dataset. Expects latents.bin + meta.npz produced
    # by preprocessing/pack_vae_latents.py. Used to bypass the NFS metadata
    # bottleneck: stage the single latents.bin into /dev/shm once, then index
    # with O(1) memmap slices in each DataLoader worker.
    _SHAPE = (8, 32, 32)
    _DTYPE = np.float32

    def __init__(self, packed_dir):
        meta = np.load(os.path.join(packed_dir, 'meta.npz'), allow_pickle=True)
        self.labels = meta['labels']
        self.n = int(meta['n'])
        shape = tuple(int(x) for x in meta['shape'])
        dtype = np.dtype(str(meta['dtype']))
        if shape != self._SHAPE or dtype != np.dtype(self._DTYPE):
            raise RuntimeError(
                f"Packed latents format mismatch: shape={shape} dtype={dtype} "
                f"(expected {self._SHAPE}, {np.dtype(self._DTYPE)})"
            )
        self._bin_path = os.path.join(packed_dir, 'latents.bin')
        # Lazy-opened per worker — memmap handles forked by DataLoader workers
        # behave best when each worker creates its own mapping.
        self._mm = None

    def _ensure_mm(self):
        if self._mm is None:
            self._mm = np.memmap(
                self._bin_path, dtype=self._DTYPE, mode='r',
                shape=(self.n,) + self._SHAPE,
            )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_mm()
        features = np.array(self._mm[idx], copy=True)
        return torch.from_numpy(features), torch.tensor(self.labels[idx])


class ImgLabelDataset(Dataset):
    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        # self.images_dir = os.path.join(data_dir, 'images')
        self.images_dir = os.path.join(data_dir, 'images')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.images_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.image_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        return torch.from_numpy(image), torch.tensor(self.labels[idx])
