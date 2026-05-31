# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import pyspng
except ImportError:
    pyspng = None

from encoders import StabilityVAEEncoder

#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

#----------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceImageEntry:
    archive_fname: str
    source_path: str
    label: Optional[int]

#----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetManifest:
    source: str
    source_type: str
    entries: tuple[SourceImageEntry, ...]

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def _default_num_workers() -> int:
    return 8

#----------------------------------------------------------------------------

def _default_write_workers() -> int:
    return 8

#----------------------------------------------------------------------------

def _load_label_map(labels_data) -> dict[str, int]:
    if labels_data is None:
        return {}
    return {fname: label for fname, label in labels_data}

#----------------------------------------------------------------------------

def _infer_labels_from_toplevel(archive_fnames: list[str]) -> dict[str, int]:
    toplevel_names = {
        archive_fname: archive_fname.split('/')[0] if '/' in archive_fname else ''
        for archive_fname in archive_fnames
    }
    toplevel_indices = {
        toplevel_name: idx
        for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))
    }
    if len(toplevel_indices) <= 1:
        return {}
    return {
        archive_fname: toplevel_indices[toplevel_name]
        for archive_fname, toplevel_name in toplevel_names.items()
    }

#----------------------------------------------------------------------------

def _scan_image_folder(source_dir: str, *, max_images: Optional[int]) -> DatasetManifest:
    input_images = []

    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file():
                    input_images.append(os.path.join(root, entry.name))
                elif entry.is_dir():
                    _recurse_dirs(os.path.join(root, entry.name))

    _recurse_dirs(source_dir)
    input_images = sorted([fname for fname in input_images if is_image_ext(fname)])
    arch_fnames = {
        fname: os.path.relpath(fname, source_dir).replace('\\', '/')
        for fname in input_images
    }

    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = _load_label_map(json.load(file)['labels'])
    if len(labels) == 0:
        labels = _infer_labels_from_toplevel(list(arch_fnames.values()))

    max_idx = maybe_min(len(input_images), max_images)
    entries = tuple(
        SourceImageEntry(
            archive_fname=arch_fnames[fname],
            source_path=fname,
            label=labels.get(arch_fnames[fname]),
        )
        for fname in input_images[:max_idx]
    )
    return DatasetManifest(source=source_dir, source_type='dir', entries=entries)

#----------------------------------------------------------------------------

def _scan_image_zip(source: str, *, max_images: Optional[int]) -> DatasetManifest:
    with zipfile.ZipFile(source, mode='r') as zf:
        input_images = [str(fname) for fname in sorted(zf.namelist()) if is_image_ext(fname)]
        labels = {}
        if 'dataset.json' in zf.namelist():
            with zf.open('dataset.json', 'r') as file:
                labels = _load_label_map(json.load(file)['labels'])
    if len(labels) == 0:
        labels = _infer_labels_from_toplevel(input_images)

    max_idx = maybe_min(len(input_images), max_images)
    entries = tuple(
        SourceImageEntry(
            archive_fname=fname,
            source_path=fname,
            label=labels.get(fname),
        )
        for fname in input_images[:max_idx]
    )
    return DatasetManifest(source=source, source_type='zip', entries=entries)

#----------------------------------------------------------------------------

def scan_dataset_manifest(source, *, max_images: Optional[int]) -> DatasetManifest:
    if os.path.isdir(source):
        return _scan_image_folder(source, max_images=max_images)
    if os.path.isfile(source):
        if file_ext(source) == 'zip':
            return _scan_image_zip(source, max_images=max_images)
        raise click.ClickException(f'Only zip archives are supported: {source}')
    raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def _decode_image_file(file_obj, fname: Union[str, Path]) -> np.ndarray:
    ext = f'.{file_ext(fname).lower()}'
    if ext == '.png' and pyspng is not None:
        raw_bytes = file_obj.read()
        img = pyspng.load(raw_bytes)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return np.array(PIL.Image.open(io.BytesIO(raw_bytes)).convert('RGB'))
    return np.array(PIL.Image.open(file_obj).convert('RGB'))

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    manifest = _scan_image_folder(source_dir, max_images=max_images)

    def iterate_images():
        for entry in manifest.entries:
            with open(entry.source_path, 'rb') as file:
                img = _decode_image_file(file, entry.source_path)
            yield ImageEntry(img=img, label=entry.label)

    return len(manifest.entries), iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    manifest = _scan_image_zip(source, max_images=max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as zf:
            for entry in manifest.entries:
                with zf.open(entry.source_path, 'r') as file:
                    img = _decode_image_file(file, entry.source_path)
                yield ImageEntry(img=img, label=entry.label)

    return len(manifest.entries), iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img, 'RGB')
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    def center_crop_imagenet(image_size: int, arr: np.ndarray):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        pil_image = PIL.Image.fromarray(arr)
        while min(*pil_image.size) >= 2 * image_size:
            new_size = tuple(x // 2 for x in pil_image.size)
            assert len(new_size) == 2
            pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    if transform == 'center-crop-dhariwal':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        if output_width != output_height:
            raise click.ClickException('width and height must match in --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_imagenet, output_width)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)

        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

def _collate_as_list(batch):
    return batch

#----------------------------------------------------------------------------

def _make_data_loader(dataset, *, batch_size: int, num_workers: Optional[int], pin_memory: bool) -> DataLoader:
    if batch_size < 1:
        raise click.ClickException('--batch-size must be at least 1')

    if num_workers is None:
        num_workers = _default_num_workers()
    if num_workers < 0:
        raise click.ClickException('Number of workers must be non-negative')

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_as_list,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True
    return DataLoader(**loader_kwargs)

#----------------------------------------------------------------------------

class _ManifestDataset(Dataset):
    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest
        self._zipfile = None

    def __len__(self):
        return len(self.manifest.entries)

    def _get_zipfile(self):
        assert self.manifest.source_type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.manifest.source, mode='r')
        return self._zipfile

    def _open_entry(self, entry: SourceImageEntry):
        if self.manifest.source_type == 'dir':
            return open(entry.source_path, 'rb')
        return self._get_zipfile().open(entry.source_path, 'r')

    def close(self):
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

#----------------------------------------------------------------------------

class _ConvertDataset(_ManifestDataset):
    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        transform: Optional[str],
        resolution: Optional[Tuple[int, int]],
    ):
        super().__init__(manifest)
        self.transform = transform
        self.output_width = resolution[0] if resolution is not None else None
        self.output_height = resolution[1] if resolution is not None else None
        self._transform = None

    def _get_transform(self):
        if self._transform is None:
            self._transform = make_transform(self.transform, self.output_width, self.output_height)
        return self._transform

    def __getstate__(self):
        return dict(super().__getstate__(), _transform=None)

    def __getitem__(self, idx):
        entry = self.manifest.entries[idx]
        with self._open_entry(entry) as file:
            img = _decode_image_file(file, entry.source_path)
        img = self._get_transform()(img)
        return {'idx': idx, 'img': img, 'label': entry.label}

#----------------------------------------------------------------------------

class _EncodeDataset(_ManifestDataset):
    def __getitem__(self, idx):
        entry = self.manifest.entries[idx]
        with self._open_entry(entry) as file:
            img = _decode_image_file(file, entry.source_path)
        img_chw = np.transpose(np.ascontiguousarray(img), (2, 0, 1))
        return {'idx': idx, 'img': img_chw, 'label': entry.label}

#----------------------------------------------------------------------------

def _png_archive_fname(idx: int) -> str:
    idx_str = f'{idx:08d}'
    return f'{idx_str[:5]}/img{idx_str}.png'

#----------------------------------------------------------------------------

def _latent_archive_fname(idx: int) -> str:
    idx_str = f'{idx:08d}'
    return f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'

#----------------------------------------------------------------------------

def _validate_image_attrs(
    dataset_attrs: Optional[dict[str, int]],
    img: np.ndarray,
    archive_fname: str,
) -> dict[str, int]:
    assert img.ndim == 3
    cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0]}
    if dataset_attrs is None:
        dataset_attrs = cur_image_attrs
        width = dataset_attrs['width']
        height = dataset_attrs['height']
        if width != height:
            raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
        if width != 2 ** int(np.floor(np.log2(width))):
            raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
    elif dataset_attrs != cur_image_attrs:
        err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
        raise click.ClickException(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))
    return dataset_attrs

#----------------------------------------------------------------------------

def _save_png(save_bytes, fname: str, img: np.ndarray) -> None:
    image_bits = io.BytesIO()
    PIL.Image.fromarray(img).save(image_bits, format='png', compress_level=0, optimize=False)
    save_bytes(fname, image_bits.getbuffer())

#----------------------------------------------------------------------------

def _save_numpy(save_bytes, fname: str, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array)
    save_bytes(fname, buffer.getvalue())

#----------------------------------------------------------------------------

def _metadata_from_labels(labels):
    return {'labels': labels if all(x is not None for x in labels) else None}

#----------------------------------------------------------------------------

def run_batched_convert(
    *,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    batch_size: int,
    num_workers: Optional[int] = None,
) -> None:
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    manifest = scan_dataset_manifest(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    dataset = _ConvertDataset(manifest, transform=transform, resolution=resolution)
    loader = _make_data_loader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    dataset_attrs = None
    labels = []

    try:
        with tqdm(total=len(manifest.entries), unit='img', smoothing=0.1) as pbar:
            for batch in loader:
                pbar.update(len(batch))
                for item in batch:
                    archive_fname = _png_archive_fname(item['idx'])
                    img = item['img']
                    if img is None:
                        continue
                    dataset_attrs = _validate_image_attrs(dataset_attrs, img, archive_fname)
                    _save_png(save_bytes, os.path.join(archive_root_dir, archive_fname), img)
                    labels.append([archive_fname, item['label']] if item['label'] is not None else None)

        save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(_metadata_from_labels(labels)))
    finally:
        close_dest()
        dataset.close()

#----------------------------------------------------------------------------

def _iter_shape_buckets(batch):
    buckets = {}
    for item in batch:
        shape = tuple(item['img'].shape)
        buckets.setdefault(shape, []).append(item)
    return buckets.values()

#----------------------------------------------------------------------------

def _encode_bucket(vae, bucket, device):
    img_batch = np.stack([item['img'] for item in bucket], axis=0)
    img_batch = torch.from_numpy(np.ascontiguousarray(img_batch))
    img_batch = img_batch.to(device, non_blocking=(device.type == 'cuda'))
    img_views = torch.cat([img_batch, torch.flip(img_batch, dims=[3])], dim=0)
    mean_std_views = vae.encode_pixels(img_views).to(torch.float32).cpu()

    encoded = []
    num_items = len(bucket)
    for pos, item in enumerate(bucket):
        encoded.append((2 * item['idx'], item['label'], mean_std_views[pos]))
        encoded.append((2 * item['idx'] + 1, item['label'], mean_std_views[num_items + pos]))
    return encoded

#----------------------------------------------------------------------------

def run_batched_encode(
    *,
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    batch_size: int,
    num_workers: Optional[int] = None,
    write_workers: Optional[int] = None,
) -> None:
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')
    if batch_size < 1:
        raise click.ClickException('--batch-size must be at least 1')

    manifest = scan_dataset_manifest(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=batch_size)
    vae.init(device)

    dataset = _EncodeDataset(manifest)
    loader = _make_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    labels = [None] * (2 * len(manifest.entries))
    dest_is_zip = file_ext(dest) == 'zip'
    if write_workers is None:
        write_workers = 0 if dest_is_zip else _default_write_workers()
    if write_workers < 0:
        raise click.ClickException('Number of write workers must be non-negative')

    pool = None
    pending = []
    max_pending = max(1, write_workers) * 4

    try:
        if not dest_is_zip and write_workers > 0:
            pool = ThreadPoolExecutor(max_workers=write_workers)

        with torch.no_grad():
            with tqdm(total=len(manifest.entries), unit='img', smoothing=0.1) as pbar:
                for batch in loader:
                    pbar.update(len(batch))
                    for bucket in _iter_shape_buckets(batch):
                        for out_idx, label, mean_std in _encode_bucket(vae, bucket, device):
                            archive_fname = _latent_archive_fname(out_idx)
                            labels[out_idx] = [archive_fname, label] if label is not None else None
                            full_fname = os.path.join(archive_root_dir, archive_fname)
                            if pool is None:
                                _save_numpy(save_bytes, full_fname, mean_std.numpy())
                            else:
                                mean_std_np = mean_std.numpy().copy()
                                pending.append(pool.submit(_save_numpy, save_bytes, full_fname, mean_std_np))
                                if len(pending) >= max_pending:
                                    pending[0].result()
                                    pending.pop(0)

        for future in pending:
            future.result()
        save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(_metadata_from_labels(labels)))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        close_dest()
        dataset.close()

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
@click.option('--batch-size', help='Number of source images to process per batch', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    batch_size: int,
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal
    """
    run_batched_convert(
        source=source,
        dest=dest,
        max_images=max_images,
        transform=transform,
        resolution=resolution,
        batch_size=batch_size,
    )

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-ema', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--batch-size', help='Number of source images to process per batch before adding flipped views', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    batch_size: int,
):
    """Encode pixel data to VAE latents."""
    run_batched_encode(
        model_url=model_url,
        source=source,
        dest=dest,
        max_images=max_images,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
