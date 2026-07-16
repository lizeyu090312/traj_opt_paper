#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/traj_opt_paper}"
LAUNCHER="$PROJECT_ROOT/final_scripts/traj_opt.sh"
MG_CKPT="${MG_CKPT:-}"

if [[ -z "$MG_CKPT" ]]; then
  echo "Set MG_CKPT to a SiT-B/2 checkpoint trained with Model Guidance." >&2
  exit 1
fi
if [[ ! -f "$MG_CKPT" ]]; then
  echo "MG checkpoint not found: $MG_CKPT" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/output"
cd "$PROJECT_ROOT"

# This wrapper changes only MG-specific inputs. The underlying three-cycle
# path-flow alignment schedule remains the same as the standard SiT-B recipe.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/output/final_repro/SiT-B-2-MG}" \
SIT_MODEL="SiT-B/2" \
PATH_MODEL="${PATH_MODEL:-SiT-B/2}" \
GEN_CONFIG_PATH="$PROJECT_ROOT/gen_configs/SiT-B-2_mg.sh" \
MODE="${MODE:-SDE}" \
SAMPLING_METHOD="${SAMPLING_METHOD:-Euler}" \
ATTN_FUNC="${ATTN_FUNC:-fa3}" \
AMP_DTYPE="${AMP_DTYPE:-bf16}" \
NUM_CYCLES="${NUM_CYCLES:-3}" \
INITIAL_FLOW_CKPT="$MG_CKPT" \
INITIAL_PATH_CKPT="$MG_CKPT" \
PATH_TEACHER_CKPT="$MG_CKPT" \
PATH_ARCH="${PATH_ARCH:-dual_stem_teacher_residual}" \
PATH_USE_ENDPOINT_CONDITIONING="${PATH_USE_ENDPOINT_CONDITIONING:-1}" \
PATH_BETA="${PATH_BETA:-0.95}" \
PATH_LEARNED_PATH_MIX="${PATH_LEARNED_PATH_MIX:-0.4}" \
FLOW_LEARNED_PATH_MIX="${FLOW_LEARNED_PATH_MIX:-0.4}" \
PATH_STAGE_STEPS="${PATH_STAGE_STEPS:-3000}" \
FLOW_STAGE_STEPS="${FLOW_STAGE_STEPS:-4000}" \
MG_START_STEP="${MG_START_STEP:-0}" \
MG_DATA_RATIO_BASE="${MG_DATA_RATIO_BASE:-0.2}" \
MG_DROP_FRAC="${MG_DROP_FRAC:-0.1}" \
MG_W_LO="${MG_W_LO:-1.45}" \
MG_W_HI="${MG_W_HI:-1.45}" \
MG_DATA_SIDE_THRESHOLD="${MG_DATA_SIDE_THRESHOLD:-0.75}" \
MG_CLASS_DROPOUT_PROB="${MG_CLASS_DROPOUT_PROB:-0.0}" \
MG_LEARN_SIGMA="${MG_LEARN_SIGMA:-0}" \
sbatch "$LAUNCHER"
