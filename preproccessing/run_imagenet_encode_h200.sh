#!/bin/bash

#SBATCH --job-name=imagenet-encode
#SBATCH --output=/path/to/traj_opt_paper/logs/%j.out
#SBATCH --error=/path/to/traj_opt_paper/logs/%j.err
#SBATCH --partition=h200
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=h200
#SBATCH --open-mode=append



set -e

cd "/path/to/traj_opt_paper"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate SiT

export PYTHONUNBUFFERED=1

export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

python preproccessing/dataset_tools.py encode \
    --model-url=stabilityai/sd-vae-ft-ema \
    --source=imagenet_SiT/images \
    --dest=imagenet_SiT/vae-sd \
    --batch-size=192
