#!/usr/bin/env bash
set -e

PROJECT_ROOT="/path/to/traj_opt_paper"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
FID_BATCH_SCRIPT="$SCRIPT_DIR/fid_from_image_dir.sh"
LOG_DIR="$SCRIPT_DIR/logs"

REF_PATH="/path/to/traj_opt_paper/data/fid_ref/VIRTUAL_imagenet256_labeled.npz"
RESULT_ROOT="/path/to/traj_opt_paper/data/fid_ref/fid_eval_final"
IMAGE_ROOT="$RESULT_ROOT/images"

COMMON_EXPORTS="FID_ONLY=0,BATCH_SIZE=200,NUM_SAMPLING_STEPS=126,SAMPLING_METHOD=heun2,NUM_FID_SAMPLES=50000,IMAGE_SIZE=256,MODE=ODE,REF_PATH=$REF_PATH,RESULT_ROOT=$RESULT_ROOT,IMAGE_ROOT=$IMAGE_ROOT"

mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$IMAGE_ROOT"

submit_fid() {
  local run_name="$1"
  local model_name="$2"
  local cfg_scale="$3"
  local checkpoint_path="$4"

  sbatch \
    --job-name="fid-${run_name}" \
    --export=ALL,$COMMON_EXPORTS,FID_RUN_NAME="$run_name",MODEL_NAME_ABBREV="$model_name",CFG_SCALE="$cfg_scale",CHECKPOINT_PATH="$checkpoint_path" \
    "$FID_BATCH_SCRIPT"
}


submit_fid "sit-xl-traj-cycle-03-0004000-ema" "SiT-XL/2" "1.7" \
  "/path/to/sit-xl-traj-cycle-03-0004000-ema.pt"

submit_fid "sit-b-traj-cycle-03-0004000-ema" "SiT-B/2" "2.5" \
  "/path/to/sit-b-traj-cycle-03-0004000-ema.pt"

submit_fid "sit-l-traj-cycle-03-0004000-ema" "SiT-L/2" "2.0" \
  "/path/to/sit-l-traj-cycle-03-0004000-ema.pt"

submit_fid "sit-s-traj-cycle-03-0004000-ema" "SiT-S/2" "3.5" \
  "/path/to/sit-s-traj-cycle-03-0004000-ema.pt"



submit_fid "sit-xl-mg-traj-cycle-03-0004000-ema" "SiT-XL/2-MG" "1.0" \
  "/path/to/sit-xl-mg-traj-cycle-03-0004000-ema.pt"

submit_fid "sit-b-mg-traj-cycle-03-0004000-ema" "SiT-B/2" "1.0" \
  "/path/to/sit-b-mg-traj-cycle-03-0004000-ema.pt"
