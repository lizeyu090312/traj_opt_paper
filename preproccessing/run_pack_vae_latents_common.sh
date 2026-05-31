#!/bin/bash

#SBATCH --job-name=pack-vae-latents
#SBATCH --output=/path/to/traj_opt_paper/logs/%j.out
#SBATCH --error=/path/to/traj_opt_paper/logs/%j.err
#SBATCH --partition=common
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --account=account
#SBATCH --open-mode=append

set -e


cd /path/to/traj_opt_paper

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate SiT

python preproccessing/pack_vae_latents.py \
    --src=imagenet_SiT/vae-sd \
    --dst=data/vae-sd-ema-packed \
    --threads=8
