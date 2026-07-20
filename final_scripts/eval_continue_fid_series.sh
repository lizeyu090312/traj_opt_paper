#!/usr/bin/env bash
set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <results_dir> [ref_dir] [num_fid_samples] [out_root] [nproc_per_node] [gen_config_path]" >&2
  exit 1
fi

RESULTS_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_DIR="${2:-$PROJECT_ROOT/data/fid_ref/VIRTUAL_imagenet256_labeled.npz}"
NUM_FID_SAMPLES="${3:-10000}"
OUT_ROOT="$4"
NPROC_PER_NODE="${5:-${NPROC_PER_NODE:-1}}"
GEN_CONFIG_PATH="${6:-${GEN_CONFIG_PATH:-}}"
GENERATED_PROJECT_ROOT="${GENERATED_PROJECT_ROOT:-/path/to/traj_opt_paper}"

if [[ -n "$GEN_CONFIG_PATH" && "$GEN_CONFIG_PATH" != /* ]]; then
  GEN_CONFIG_PATH="$PROJECT_ROOT/$GEN_CONFIG_PATH"
fi
if [[ -n "$GEN_CONFIG_PATH" ]]; then
  if [[ ! -f "$GEN_CONFIG_PATH" ]]; then
    echo "Generation config not found at $GEN_CONFIG_PATH" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$GEN_CONFIG_PATH"
fi

MODEL="${MODEL:-SiT-S/2}"
ATTN_FUNC="${ATTN_FUNC:-}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CFG_SCALE="${CFG_SCALE:-4.0}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-100}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
BASELINE_FID="${BASELINE_FID:-}"
MODE="${MODE:-ODE}"
SAMPLING_METHOD="${SAMPLING_METHOD:-}"
NUM_SAMPLING_NFE="${NUM_SAMPLING_NFE:-}"

if [[ "$MODE" != "ODE" && "$MODE" != "SDE" ]]; then
  echo "MODE must be ODE or SDE, got '$MODE'." >&2
  exit 1
fi

# torchdiffeq's fixed-grid Heun2 evaluates the model twice per interval. The
# sampler argument is the number of grid points, so N NFE needs N/2 + 1 points.
if [[ -n "$NUM_SAMPLING_NFE" ]]; then
  if [[ "$MODE" != "ODE" || "$SAMPLING_METHOD" != "heun2" ]]; then
    echo "NUM_SAMPLING_NFE is currently supported only for MODE=ODE and SAMPLING_METHOD=heun2." >&2
    exit 1
  fi
  if [[ ! "$NUM_SAMPLING_NFE" =~ ^[1-9][0-9]*$ ]] || (( NUM_SAMPLING_NFE % 2 != 0 )); then
    echo "Heun2 NUM_SAMPLING_NFE must be a positive even integer, got '$NUM_SAMPLING_NFE'." >&2
    exit 1
  fi
  NUM_SAMPLING_STEPS=$((NUM_SAMPLING_NFE / 2 + 1))
  echo "Heun2 evaluation: ${NUM_SAMPLING_NFE} NFE = ${NUM_SAMPLING_STEPS} solver time points."
fi

mkdir -p "$OUT_ROOT"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_ROOT/.mplconfig}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$MPLCONFIGDIR"

mapfile -t EMA_CKPTS < <(find "$RESULTS_DIR" -path '*/checkpoints/*_ema*.pt' -type f | sort)
if [[ "${#EMA_CKPTS[@]}" -eq 0 ]]; then
  echo "No EMA checkpoints found under $RESULTS_DIR" >&2
  exit 1
fi

RESULT_TSV="${RESULT_TSV:-$OUT_ROOT/fid_regression.tsv}"
if [[ ! -f "$RESULT_TSV" ]]; then
  echo -e "checkpoint\tfid\tcfg" | tee "$RESULT_TSV"
fi

prepare_generated_root() {
  local run_dir="$1"
  local generated_root="$run_dir/generated"
  if [[ "$generated_root" == "$PROJECT_ROOT/output" || "$generated_root" == "$PROJECT_ROOT/output/"* ]]; then
    local mirrored_root="${GENERATED_PROJECT_ROOT}${generated_root#$PROJECT_ROOT}"
    mkdir -p "$mirrored_root"
    if [[ -L "$generated_root" ]]; then
      ln -sfnT "$mirrored_root" "$generated_root"
    elif [[ ! -e "$generated_root" ]]; then
      ln -sT "$mirrored_root" "$generated_root"
    fi
  else
    mkdir -p "$generated_root"
  fi
  printf '%s\n' "$generated_root"
}

find_latest_generated_dir() {
  local generated_root="$1"
  if [[ -d "$generated_root" ]]; then
    find -L "$generated_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
  fi
}

for CKPT in "${EMA_CKPTS[@]}"; do
  TAG="$(basename "$CKPT" .pt)"
  RUN_DIR="$OUT_ROOT/$TAG"
  mkdir -p "$RUN_DIR"

  if [[ -f "$RUN_DIR/fid.txt" ]]; then
    FID_VALUE="$(awk '/^FID:/ {print $2}' "$RUN_DIR/fid.txt" | tail -n 1)"
    if [[ -n "$FID_VALUE" ]]; then
      echo "Skipping $CKPT because fid.txt already exists."
      grep -q "^${TAG}[[:space:]]" "$RESULT_TSV" || echo -e "${TAG}\t${FID_VALUE}\t${CFG_SCALE}" | tee -a "$RESULT_TSV"
      continue
    fi
  fi

  echo "Evaluating $CKPT"
  GENERATED_ROOT="$(prepare_generated_root "$RUN_DIR")"
  GEN_DIR=""

  sample_attn_args=()
  if [[ -n "$ATTN_FUNC" ]]; then
    sample_attn_args+=(--attn-func "$ATTN_FUNC")
  fi
  sample_method_args=()
  if [[ -n "$SAMPLING_METHOD" ]]; then
    sample_method_args+=(--sampling-method "$SAMPLING_METHOD")
  fi
  "$ENV_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" sample_ddp.py "$MODE" \
    --model "$MODEL" \
    "${sample_attn_args[@]}" \
    "${sample_method_args[@]}" \
    --image-size "$IMAGE_SIZE" \
    --ckpt "$CKPT" \
    --vae ema \
    --cfg-scale "$CFG_SCALE" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --no-make-npz \
    --sample-dir "$GENERATED_ROOT"
  GEN_DIR="$(find_latest_generated_dir "$GENERATED_ROOT")"

  if [[ -z "${GEN_DIR:-}" ]]; then
    echo "No generated image directory found for $CKPT" >&2
    exit 1
  fi

  FID_OUTPUT="$("$ENV_PYTHON" -m pytorch_fid "$REF_DIR" "$GEN_DIR" --device cuda:0)"
  printf "%s\n" "$FID_OUTPUT" | tee "$RUN_DIR/fid.txt"
  FID_VALUE="$(printf "%s\n" "$FID_OUTPUT" | awk '/^FID:/ {print $2}' | tail -n 1)"
  grep -q "^${TAG}[[:space:]]" "$RESULT_TSV" || echo -e "${TAG}\t${FID_VALUE}\t${CFG_SCALE}" | tee -a "$RESULT_TSV"
done
