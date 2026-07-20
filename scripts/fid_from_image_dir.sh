#!/usr/bin/env bash

#SBATCH --job-name=fid
#SBATCH --output=/path/to/traj_opt_paper/scripts/logs/%x-%j.out
#SBATCH --error=/path/to/traj_opt_paper/scripts/logs/%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=h200
#SBATCH --open-mode=append

set -e

PROJECT_ROOT="/path/to/traj_opt_paper"
cd "$PROJECT_ROOT"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate SiT

export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# Sampling / FID config. Edit these values directly when you want a different setup.
MODEL_NAME_ABBREV="${MODEL_NAME_ABBREV:-SiT-B/2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MODE="${MODE:-ODE}"
CFG_SCALE="${CFG_SCALE:-2.0}"
BATCH_SIZE="${BATCH_SIZE:-250}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-126}"
SAMPLING_METHOD="${SAMPLING_METHOD:-heun2}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
FID_ONLY="${FID_ONLY:-0}"  # Set to 1/true to compute FID from existing images.

CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH must be set to a single .pt checkpoint}"
FID_RUN_NAME="${FID_RUN_NAME:?FID_RUN_NAME must be set for this sbatch}"
REF_PATH="${REF_PATH:-/path/to/traj_opt_paper/data/fid_ref/VIRTUAL_imagenet256_labeled.npz}"
RESULT_ROOT="${RESULT_ROOT:-/path/to/traj_opt_paper/data/fid_ref/fid_eval_final}"
IMAGE_ROOT="${IMAGE_ROOT:-$RESULT_ROOT/images}"

# Relative path inputs are resolved under PROJECT_ROOT. The driver passes absolute
# paths for CHECKPOINT_PATH, REF_PATH, RESULT_ROOT, IMAGE_ROOT, and RESULT_TSV.
project_abspath() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$path"
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

CHECKPOINT_PATH="$(project_abspath "$CHECKPOINT_PATH")"
REF_PATH="$(project_abspath "$REF_PATH")"
RESULT_ROOT="$(project_abspath "$RESULT_ROOT")"
IMAGE_ROOT="$(project_abspath "$IMAGE_ROOT")"

CHECKPOINT_TAG="${CHECKPOINT_TAG:-$(basename "$CHECKPOINT_PATH" .pt)}"
RUN_ROOT="$RESULT_ROOT/runs/$FID_RUN_NAME"
SAMPLE_PARENT_DIR="${SAMPLE_PARENT_DIR:-$IMAGE_ROOT/${FID_RUN_NAME}_img${NUM_FID_SAMPLES}}"
SAMPLE_PARENT_DIR="$(project_abspath "$SAMPLE_PARENT_DIR")"
RESULT_TSV="${RESULT_TSV:-$RUN_ROOT/fid_result.tsv}"
RESULT_TSV="$(project_abspath "$RESULT_TSV")"

build_sample_folder_name() {
  printf '%s-%s-cfg-%s-%s-%s-%s-%s\n' \
    "${MODEL_NAME_ABBREV//\//-}" \
    "$CHECKPOINT_TAG" \
    "$CFG_SCALE" \
    "$BATCH_SIZE" \
    "$MODE" \
    "$NUM_SAMPLING_STEPS" \
    "$SAMPLING_METHOD"
}

SAMPLE_FOLDER_NAME="${SAMPLE_FOLDER_NAME:-$(build_sample_folder_name)}"
GENERATED_IMAGE_DIR="$SAMPLE_PARENT_DIR/$SAMPLE_FOLDER_NAME"

print_run_config() {
  cat <<EOF
===== FID evaluation config =====
project-root: $PROJECT_ROOT
fid-run-name: $FID_RUN_NAME
model-name-abbrev: $MODEL_NAME_ABBREV
image-size: $IMAGE_SIZE
mode: $MODE
cfg-scale: $CFG_SCALE
batch-size: $BATCH_SIZE
num-sampling-steps: $NUM_SAMPLING_STEPS
sampling-method: $SAMPLING_METHOD
num-fid-samples: $NUM_FID_SAMPLES
fid-only: $FID_ONLY
ref-path: $REF_PATH
checkpoint-path: $CHECKPOINT_PATH
sample-parent-dir: $SAMPLE_PARENT_DIR
sample-folder-name: $SAMPLE_FOLDER_NAME
generated-image-path: $GENERATED_IMAGE_DIR
result-tsv: $RESULT_TSV
=================================
EOF
}

generate_if_needed() {
  mkdir -p "$SAMPLE_PARENT_DIR" "$RUN_ROOT" "$(dirname "$RESULT_TSV")"

  if is_truthy "$FID_ONLY"; then
    return 0
  fi

  if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
  fi

  python -m torch.distributed.run --standalone --nproc_per_node=1 sample_ddp.py "$MODE" \
    --model "$MODEL_NAME_ABBREV" \
    --image-size "$IMAGE_SIZE" \
    --ckpt "$CHECKPOINT_PATH" \
    --vae ema \
    --cfg-scale "$CFG_SCALE" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" \
    --sampling-method "$SAMPLING_METHOD" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --no-make-npz \
    --sample-dir "$SAMPLE_PARENT_DIR" \
    --sample-folder-name "$SAMPLE_FOLDER_NAME" \
    --per-proc-batch-size "$BATCH_SIZE"
}

write_fid() {
  local fid_text
  local fid

  if [[ ! -d "$GENERATED_IMAGE_DIR" ]]; then
    echo "Sample directory not found: $GENERATED_IMAGE_DIR" >&2
    exit 1
  fi

  fid_text="$(python -m pytorch_fid "$REF_PATH" "$GENERATED_IMAGE_DIR" --device cuda:0)"
  printf '%s\n' "$fid_text"
  fid="$(printf '%s\n' "$fid_text" | awk '/^FID:/ {print $2}' | tail -n 1)"

  if [[ -z "$fid" ]]; then
    echo "Could not parse FID from pytorch_fid output." >&2
    exit 1
  fi

  if [[ ! -f "$RESULT_TSV" ]]; then
    printf "run_name\tcheckpoint_path\tfid\tgenerated_image_path\tmodel\tcfg_scale\tbatch_size\tnum_sampling_steps\tsampling_method\tnum_fid_samples\n" > "$RESULT_TSV"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$FID_RUN_NAME" \
    "$CHECKPOINT_PATH" \
    "$fid" \
    "$GENERATED_IMAGE_DIR" \
    "$MODEL_NAME_ABBREV" \
    "$CFG_SCALE" \
    "$BATCH_SIZE" \
    "$NUM_SAMPLING_STEPS" \
    "$SAMPLING_METHOD" \
    "$NUM_FID_SAMPLES" >> "$RESULT_TSV"

  echo "===== Final FID ====="
  printf 'FID: %s\n' "$fid"
  printf 'checkpoint-path: %s\n' "$CHECKPOINT_PATH"
  printf 'generated-image-path: %s\n' "$GENERATED_IMAGE_DIR"
}

print_run_config
generate_if_needed
write_fid
