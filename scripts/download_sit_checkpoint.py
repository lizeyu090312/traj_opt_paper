#!/usr/bin/env python3
import argparse
import os
import shutil

from huggingface_hub import hf_hub_download


MODEL_TO_FILE = {
    "SiT-S/2": "SiT-S-2-256_orig.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SiT-S/2", choices=sorted(MODEL_TO_FILE.keys()))
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    filename = MODEL_TO_FILE[args.model]
    cache_path = hf_hub_download(repo_id="nyu-visionx/SiT-collections", filename=filename,)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, filename)
    if os.path.abspath(cache_path) != os.path.abspath(out_path):
        shutil.copy2(cache_path, out_path)

    print(out_path)


if __name__ == "__main__":
    main()
