#!/usr/bin/env python3
"""Thin wrapper around the shared batched VAE encoder in dataset_tools.py.

This keeps the more configurable CLI for ad hoc preprocessing runs while
reusing the same encode implementation and output semantics as
`dataset_tools.py encode`, including writing both the original and
horizontally flipped latent for each source image.
"""

import argparse

from dataset_tools import run_batched_encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Input directory or archive name')
    parser.add_argument('--dest', required=True, help='Output directory or archive name')
    parser.add_argument('--batch-size', type=int, default=64, help='Number of source images per batch before adding flipped views')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker processes')
    parser.add_argument('--write-workers', type=int, default=4, help='Concurrent folder write workers')
    parser.add_argument('--vae-name', default='stabilityai/sd-vae-ft-ema', help='VAE model identifier')
    parser.add_argument('--max-images', type=int, default=None, help='Optional cap on source images to encode')
    args = parser.parse_args()

    run_batched_encode(
        model_url=args.vae_name,
        source=args.source,
        dest=args.dest,
        max_images=args.max_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        write_workers=args.write_workers,
    )


if __name__ == '__main__':
    main()
