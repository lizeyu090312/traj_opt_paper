This directory contains the ImageNet preprocessing pipeline used for this repo.
It is adapted from [edm2](https://github.com/NVlabs/edm2). The workflow is intended for 256x256 experiments.

1. `dataset_tools.py convert`
   Converts raw ImageNet images into a resized PNG dataset in `images/`.
2. `dataset_tools.py encode`
   Encodes those PNGs into per-sample Stable Diffusion VAE latent files in `vae-sd/`.
3. `pack_vae_latents.py`
   Packs the per-sample `.npy` latents into a single `latents.bin` plus `meta.npz` in `vae-sd-ema-packed/`.

The directory `data/vae-sd-ema-packed/` contains `latents.bin` and `meta.npz`, where `data/vae-sd-ema-packed/` is produced by `pack_vae_latents.py` from an existing `vae-sd/` directory.


Run the commands below.

```bash
# 1) resize the images
python preproccessing/dataset_tools.py convert \
    --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images \
    --batch-size=32 \
    --resolution=256x256 \
    --transform=center-crop-dhariwal
```

```bash
# 2) encode the PNG dataset into latents
python preproccessing/dataset_tools.py encode \
    --source=[TARGET_PATH]/images --batch-size=8 --dest=[TARGET_PATH]/vae-sd
```

```bash
# 3) pack the latents
python preproccessing/pack_vae_latents.py --src=[TARGET_PATH]/vae-sd --dst=[TARGET_PATH]/vae-sd-ema-packed --threads=64
```

- `YOUR_DOWNLOAD_PATH` is the location of the raw ImageNet download.
- `TARGET_PATH` is the directory where the processed artifacts will be written, set this to be `./data`. 

After these steps, your directory will typically look like:

```text
[TARGET_PATH]/
  images/          resized PNGs, dataset.json
  vae-sd/          latent .npy files + dataset.json
  vae-sd-ema-packed/   latents.bin and meta.npz
```

This repository uses `vae-sd-ema-packed/`. The training scripts detect the packed format by checking for `latents.bin` and `meta.npz`, so `vae-sd-ema-packed/` can be passed directly as `--data-path`.
