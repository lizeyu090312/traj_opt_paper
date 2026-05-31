#!/bin/bash

#SBATCH --job-name=imagenet-convert
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

PROJECT_ROOT="/path/to/traj_opt_paper"
cd "$PROJECT_ROOT"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate SiT

export CUDA_VISIBLE_DEVICES=""

python preproccessing/dataset_tools.py convert \
    --source="/path/to/imagenet_ILSVRC2012/train" \
    --dest="imagenet_SiT/images" \
    --batch-size="256" --resolution=256x256 \
    --transform=center-crop-dhariwal
