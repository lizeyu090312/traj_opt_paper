#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAUNCHER="$PROJECT_ROOT/final_scripts/traj_opt.sh"
MG_CKPT="$PROJECT_ROOT/finetuned_ckpt/sit-xl-2-mg_initial.pt"

if [[ ! -f "$MG_CKPT" ]]; then
  echo "MG checkpoint not found: $MG_CKPT" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/finetuned_ckpt/sit-b-2-mg_initial.pt" ]]; then
  echo "MG checkpoint not found: $PROJECT_ROOT/finetuned_ckpt/sit-b-2-mg_initial.pt" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/output"
cd "$PROJECT_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/output/final_repro/SiT-XL-2-MG}" \
SIT_MODEL="SiT-XL/2-MG" \
PATH_MODEL="${PATH_MODEL:-SiT-B/2}" \
GEN_CONFIG_PATH="$PROJECT_ROOT/gen_configs/SiT-XL-2_mg.sh" \
ATTN_FUNC="${ATTN_FUNC:-fa3}" \
AMP_DTYPE="${AMP_DTYPE:-bf16}" \
NUM_CYCLES="${NUM_CYCLES:-3}" \
INITIAL_FLOW_CKPT="$MG_CKPT" \
INITIAL_PATH_CKPT="$PROJECT_ROOT/finetuned_ckpt/sit-b-2-mg_initial.pt" \
PATH_TEACHER_CKPT="$MG_CKPT" \
PATH_ARCH="${PATH_ARCH:-dual_stem_teacher_residual}" \
PATH_USE_ENDPOINT_CONDITIONING="${PATH_USE_ENDPOINT_CONDITIONING:-1}" \
PATH_BETA="${PATH_BETA:-0.95}" \
PATH_LEARNED_PATH_MIX="${PATH_LEARNED_PATH_MIX:-0.4}" \
FLOW_LEARNED_PATH_MIX="${FLOW_LEARNED_PATH_MIX:-0.4}" \
X0_HAT_RHO_SCALE="${X0_HAT_RHO_SCALE:-0.7}" \
PATH_GLOBAL_BATCH_SIZE="${PATH_GLOBAL_BATCH_SIZE:-140}" \
PATH_PER_GPU_BATCH_SIZE="${PATH_PER_GPU_BATCH_SIZE:-70}" \
FLOW_PER_GPU_BATCH_SIZE="${FLOW_PER_GPU_BATCH_SIZE:-192}" \
FLOW_EMA_DECAY="${FLOW_EMA_DECAY:-0.999}" \
PATH_LR="${PATH_LR:-5e-5}" \
PATH_MIN_LR="${PATH_MIN_LR:-5e-6}" \
PATH_STAGE_STEPS="${PATH_STAGE_STEPS:-3000}" \
PATH_CKPT_EVERY="${PATH_CKPT_EVERY:-3000}" \
PATH_EVAL_EVERY="${PATH_EVAL_EVERY:-3000}" \
FLOW_STAGE_STEPS="${FLOW_STAGE_STEPS:-4000}" \
MG_START_STEP="${MG_START_STEP:-0}" \
sbatch "$LAUNCHER"
