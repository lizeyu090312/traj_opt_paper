"""Pack 1.28M per-sample VAE latents into one fixed-stride binary.

Eliminates the NFS metadata-ops bottleneck: training then reads one sequential
~40 GB file into /dev/shm instead of opening 1.28M individual .npy files.

Artifacts written to --dst:
    latents.bin   raw fp32, shape (N, 8, 32, 32), size N*32768 bytes
    meta.npz      filenames (object), labels, n, shape, dtype

Filename ordering mirrors dataset.VAELabelDataset.__init__ exactly so the
label index remains aligned with the legacy dataset.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SHAPE = (8, 32, 32)
DTYPE = np.float32
BYTES_PER_ITEM = int(np.prod(SHAPE)) * np.dtype(DTYPE).itemsize  # 32768


def collect_feature_fnames(features_dir):
    fnames = {
        os.path.relpath(os.path.join(root, fname), start=features_dir)
        for root, _dirs, files in os.walk(features_dir) for fname in files
    }
    return sorted(f for f in fnames if f.lower().endswith('.npy'))


def load_labels(features_dir, feature_fnames):
    with open(os.path.join(features_dir, 'dataset.json'), 'rb') as f:
        labels_raw = json.load(f)['labels']
    labels_map = dict(labels_raw)
    labels = [labels_map[f.replace('\\', '/')] for f in feature_fnames]
    labels = np.array(labels)
    return labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


def validate_first(features_dir, feature_fnames):
    probe = np.load(os.path.join(features_dir, feature_fnames[0]))
    if probe.shape != SHAPE or probe.dtype != DTYPE:
        raise RuntimeError(
            f"Unexpected latent format: shape={probe.shape} dtype={probe.dtype} "
            f"(expected shape={SHAPE} dtype={DTYPE})"
        )
    if probe.tobytes().__len__() != BYTES_PER_ITEM:
        raise RuntimeError("Payload size mismatch")


def _write_one(out_fd, features_dir, fname, idx):
    arr = np.load(os.path.join(features_dir, fname))
    if arr.shape != SHAPE or arr.dtype != DTYPE:
        raise RuntimeError(f"shape/dtype drift at {fname}: {arr.shape} {arr.dtype}")
    buf = np.ascontiguousarray(arr, dtype=DTYPE).tobytes()
    # pwrite is atomic w.r.t. offset — safe to share one fd across threads
    os.pwrite(out_fd, buf, idx * BYTES_PER_ITEM)


def pack(features_dir, dst_dir, threads):
    os.makedirs(dst_dir, exist_ok=True)
    feature_fnames = collect_feature_fnames(features_dir)
    n = len(feature_fnames)
    print(f"Found {n:,} latent files under {features_dir}", flush=True)
    if n == 0:
        raise SystemExit("No .npy files found")

    validate_first(features_dir, feature_fnames)
    labels = load_labels(features_dir, feature_fnames)
    print(f"Labels loaded: shape={labels.shape} dtype={labels.dtype}", flush=True)

    bin_path = os.path.join(dst_dir, 'latents.bin')
    total_bytes = n * BYTES_PER_ITEM
    print(f"Preallocating {bin_path} ({total_bytes / 2**30:.2f} GiB)", flush=True)
    with open(bin_path, 'wb') as f:
        f.truncate(total_bytes)

    fd = os.open(bin_path, os.O_WRONLY)
    t0 = time.time()
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = {
                ex.submit(_write_one, fd, features_dir, fname, i): i
                for i, fname in enumerate(feature_fnames)
            }
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50000 == 0 or done == n:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n - done) / rate if rate > 0 else float('inf')
                    print(
                        f"  packed {done:,}/{n:,} "
                        f"({rate:,.0f} files/s, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                        flush=True,
                    )
    finally:
        os.fsync(fd)
        os.close(fd)

    meta_path = os.path.join(dst_dir, 'meta.npz')
    np.savez(
        meta_path,
        filenames=np.array(feature_fnames, dtype=object),
        labels=labels,
        n=np.int64(n),
        shape=np.array(SHAPE, dtype=np.int64),
        dtype=np.array(str(np.dtype(DTYPE)), dtype=object),
    )
    print(f"Wrote {meta_path}", flush=True)

    verify(features_dir, dst_dir, feature_fnames, n)


def verify(features_dir, dst_dir, feature_fnames, n):
    print("Verifying 128 random samples against source .npy files...", flush=True)
    mm = np.memmap(
        os.path.join(dst_dir, 'latents.bin'),
        dtype=DTYPE, mode='r', shape=(n,) + SHAPE,
    )
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=128, replace=False)
    # Always include the endpoints
    sample_idx = np.unique(np.concatenate([sample_idx, [0, n - 1]]))
    for idx in sample_idx:
        expected = np.load(os.path.join(features_dir, feature_fnames[idx]))
        got = np.asarray(mm[idx])
        if not np.array_equal(expected, got):
            raise RuntimeError(f"Mismatch at idx={idx} ({feature_fnames[idx]})")
    print(f"OK — {len(sample_idx)} samples match", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='vae-sd dir with per-bucket .npy files and dataset.json')
    ap.add_argument('--dst', required=True, help='output dir for latents.bin + meta.npz')
    ap.add_argument('--threads', type=int, default=64)
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.src, 'dataset.json')):
        print(f"ERROR: {args.src}/dataset.json not found", file=sys.stderr)
        sys.exit(1)

    pack(args.src, args.dst, args.threads)


if __name__ == '__main__':
    main()
