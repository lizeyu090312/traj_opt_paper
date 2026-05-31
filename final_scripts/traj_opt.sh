#!/bin/bash

#SBATCH --job-name=h200
#SBATCH --output=/path/to/traj_opt_paper/logs/%j.out
#SBATCH --error=/path/to/traj_opt_paper/logs/%j.err
#SBATCH --partition=h200
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:2
#SBATCH --account=h200
#SBATCH --open-mode=append

set -e

PROJECT_ROOT="/path/to/traj_opt_paper"
cd $PROJECT_ROOT

CONDA_ENV_NAME="${CONDA_ENV_NAME:-SiT}"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

ENV_PYTHON="${ENV_PYTHON:-$(command -v python)}"
IMAGENET_TRAIN="$PROJECT_ROOT/data/vae-sd-ema-packed"

REF_DIR="${REF_DIR:-$PROJECT_ROOT/data/fid_ref/VIRTUAL_imagenet256_labeled.npz}"
SIT_MODEL="${SIT_MODEL:-SiT-S/2}"
PATH_MODEL="${PATH_MODEL:-$SIT_MODEL}"
ATTN_FUNC="${ATTN_FUNC:-fa3}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
PACKED_LATENT_VIEW_MODE="${PACKED_LATENT_VIEW_MODE:-auto}"

default_gen_config_path() {
  case "$1" in
    "SiT-S/2") printf '%s\n' "$PROJECT_ROOT/gen_configs/SiT-S-2.sh" ;;
    "SiT-B/2") printf '%s\n' "$PROJECT_ROOT/gen_configs/SiT-B-2.sh" ;;
    *) return 1 ;;
  esac
}

GEN_CONFIG_PATH="${GEN_CONFIG_PATH:-$(default_gen_config_path "$SIT_MODEL" || true)}"
if [[ -z "$GEN_CONFIG_PATH" ]]; then
  echo "Set GEN_CONFIG_PATH for unsupported SIT_MODEL '$SIT_MODEL'." >&2
  exit 1
fi
if [[ ! -f "$GEN_CONFIG_PATH" ]]; then
  echo "Generation config not found at $GEN_CONFIG_PATH" >&2
  exit 1
fi

INITIAL_FLOW_CKPT="${INITIAL_FLOW_CKPT:-$PROJECT_ROOT/checkpoints/SiT-S-2-256_orig.pt}"
INITIAL_PATH_CKPT="${INITIAL_PATH_CKPT:-$PROJECT_ROOT/checkpoints/SiT-S-2-256_orig.pt}"

OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/output/results_altmin_cycles_h200}"
RUN_DIR="${OUT_ROOT}/job_${SLURM_JOB_ID}"

NUM_CYCLES="${NUM_CYCLES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PATH_GLOBAL_BATCH_SIZE="${PATH_GLOBAL_BATCH_SIZE:-280}"
PATH_PER_GPU_BATCH_SIZE="${PATH_PER_GPU_BATCH_SIZE:-$PATH_GLOBAL_BATCH_SIZE}"
FLOW_GLOBAL_BATCH_SIZE="${FLOW_GLOBAL_BATCH_SIZE:-768}"
FLOW_PER_GPU_BATCH_SIZE="${FLOW_PER_GPU_BATCH_SIZE:-$FLOW_GLOBAL_BATCH_SIZE}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BASELINE_FID="${BASELINE_FID:-17.322351396265162}"

PATH_BETA="${PATH_BETA:-0.9}"
PATH_FD_STEP="${PATH_FD_STEP:-0.02}"
PATH_ACCEL_REG_WEIGHT="${PATH_ACCEL_REG_WEIGHT:-0.0}"
PATH_LOSS_MODE="${PATH_LOSS_MODE:-default}"
PATH_LEARNED_PATH_MIX="${PATH_LEARNED_PATH_MIX:-1.0}"
X0_HAT_RHO_SCALE="${X0_HAT_RHO_SCALE:-1}"
PATH_LR="${PATH_LR:-3e-4}"
PATH_MIN_LR="${PATH_MIN_LR:-3e-5}"
PATH_LR_WARMUP_STEPS="${PATH_LR_WARMUP_STEPS:-50}"
PATH_LR_ANNEAL_STEPS="${PATH_LR_ANNEAL_STEPS:-3000}"
PATH_WEIGHT_DECAY="${PATH_WEIGHT_DECAY:-0.0}"
PATH_NET_EMA="${PATH_NET_EMA:-0}"
PATH_STAGE_STEPS="${PATH_STAGE_STEPS:-3000}"
PATH_LOG_EVERY="${PATH_LOG_EVERY:-25}"
PATH_CKPT_EVERY="${PATH_CKPT_EVERY:-100}"
PATH_AUTOSAVE_EVERY="${PATH_AUTOSAVE_EVERY:-200}"
PATH_EVAL_EVERY="${PATH_EVAL_EVERY:-100}"
PATH_EVAL_NUM_BATCHES="${PATH_EVAL_NUM_BATCHES:-4}"
PATH_ARCH="${PATH_ARCH:-legacy_sit}"
PATH_USE_ENDPOINT_CONDITIONING="${PATH_USE_ENDPOINT_CONDITIONING:-1}"
SKIP_PATH_STAGE_CYCLE1="${SKIP_PATH_STAGE_CYCLE1:-0}"
# Cycle 1 only; later cycles reuse the previous flow EMA as the path teacher.
PATH_TEACHER_CKPT="${PATH_TEACHER_CKPT:-$PROJECT_ROOT/checkpoints/SiT-S-2-256.pt}"

FLOW_LEARNED_PATH_MIX="${FLOW_LEARNED_PATH_MIX:-0.1}"
FLOW_LEARNED_PATH_SUBTRACTOR_RESIDUAL_SCALE="${FLOW_LEARNED_PATH_SUBTRACTOR_RESIDUAL_SCALE:-1.0}"
FLOW_LR="${FLOW_LR:-5e-6}"
FLOW_WEIGHT_DECAY="${FLOW_WEIGHT_DECAY:-0.0}"
FLOW_EMA_DECAY="${FLOW_EMA_DECAY:-0.9999}"
FLOW_STAGE_STEPS="${FLOW_STAGE_STEPS:-2000}"
FLOW_LOG_EVERY="${FLOW_LOG_EVERY:-50}"
FLOW_CKPT_EVERY="${FLOW_CKPT_EVERY:-2000}"
FLOW_AUTOSAVE_EVERY="${FLOW_AUTOSAVE_EVERY:-200}"
FLOW_RESUME_CKPT="${FLOW_RESUME_CKPT:-}"
RESUME_FROM_CYCLE_DIR="${RESUME_FROM_CYCLE_DIR:-}"

path_residual_x0_time_rho_args=()
if [[ "${DISABLE_PATH_RESIDUAL_X0_TIME_RHO:-0}" == "1" ]]; then
  path_residual_x0_time_rho_args=(--disable-path-residual-x0-time-rho)
fi

GPU_LOG_PIDS=()

cleanup() {
  for pid in "${GPU_LOG_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}

start_gpu_log() {
  local dest="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 15 > "$dest" &
  GPU_LOG_PIDS+=($!)
}

stop_last_gpu_log() {
  local count=${#GPU_LOG_PIDS[@]}
  if [[ "$count" -gt 0 ]]; then
    local last_index=$((count - 1))
    local pid="${GPU_LOG_PIDS[$last_index]}"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    unset 'GPU_LOG_PIDS[$last_index]'
  fi
}

trap cleanup EXIT

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/output" "$OUT_ROOT" "$RUN_DIR"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLCONFIGDIR="$RUN_DIR/.mplconfig"
export HF_HOME="$PROJECT_ROOT/tmp/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PROJECT_ROOT/tmp/torch"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$MPLCONFIGDIR" "$HF_HUB_CACHE" "$TORCH_HOME"

export SSL_CERT_FILE="$($ENV_PYTHON -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export ENV_PYTHON
export BASELINE_FID

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "ENV_PYTHON=$ENV_PYTHON"
echo "RUN_DIR=$RUN_DIR"
echo "INITIAL_FLOW_CKPT=$INITIAL_FLOW_CKPT"
echo "INITIAL_PATH_CKPT=$INITIAL_PATH_CKPT"
echo "IMAGENET_TRAIN=$IMAGENET_TRAIN"
echo "REF_DIR=$REF_DIR"
echo "NUM_CYCLES=$NUM_CYCLES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PATH_GLOBAL_BATCH_SIZE=$PATH_GLOBAL_BATCH_SIZE"
echo "PATH_PER_GPU_BATCH_SIZE=$PATH_PER_GPU_BATCH_SIZE"
echo "FLOW_GLOBAL_BATCH_SIZE=$FLOW_GLOBAL_BATCH_SIZE"
echo "FLOW_PER_GPU_BATCH_SIZE=$FLOW_PER_GPU_BATCH_SIZE"
echo "NUM_FID_SAMPLES=$NUM_FID_SAMPLES"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "SIT_MODEL=$SIT_MODEL"
echo "PATH_MODEL=$PATH_MODEL"
echo "GEN_CONFIG_PATH=$GEN_CONFIG_PATH"
echo "ATTN_FUNC=$ATTN_FUNC"
echo "AMP_DTYPE=$AMP_DTYPE"
echo "PACKED_LATENT_VIEW_MODE=$PACKED_LATENT_VIEW_MODE"
echo "DISABLE_PATH_RESIDUAL_X0_TIME_RHO=${DISABLE_PATH_RESIDUAL_X0_TIME_RHO:-0}"
echo "PATH_BETA=$PATH_BETA"
echo "PATH_ACCEL_REG_WEIGHT=$PATH_ACCEL_REG_WEIGHT"
echo "PATH_LOSS_MODE=$PATH_LOSS_MODE"
echo "PATH_LEARNED_PATH_MIX=$PATH_LEARNED_PATH_MIX"
echo "X0_HAT_RHO_SCALE=$X0_HAT_RHO_SCALE"
echo "PATH_LR=$PATH_LR"
echo "PATH_MIN_LR=$PATH_MIN_LR"
echo "PATH_NET_EMA=$PATH_NET_EMA"
echo "PATH_STAGE_STEPS=$PATH_STAGE_STEPS"
echo "PATH_AUTOSAVE_EVERY=$PATH_AUTOSAVE_EVERY"
echo "PATH_ARCH=$PATH_ARCH"
echo "PATH_USE_ENDPOINT_CONDITIONING=$PATH_USE_ENDPOINT_CONDITIONING"
echo "SKIP_PATH_STAGE_CYCLE1=$SKIP_PATH_STAGE_CYCLE1"
echo "PATH_TEACHER_CKPT (cycle 1 only)=$PATH_TEACHER_CKPT"
echo "FLOW_LEARNED_PATH_MIX=$FLOW_LEARNED_PATH_MIX"
echo "FLOW_LEARNED_PATH_SUBTRACTOR_RESIDUAL_SCALE=$FLOW_LEARNED_PATH_SUBTRACTOR_RESIDUAL_SCALE"
echo "FLOW_STAGE_STEPS=$FLOW_STAGE_STEPS"
echo "FLOW_CKPT_EVERY=$FLOW_CKPT_EVERY"
echo "FLOW_AUTOSAVE_EVERY=$FLOW_AUTOSAVE_EVERY"
echo "FLOW_LR=$FLOW_LR"
echo "FLOW_EMA_DECAY=$FLOW_EMA_DECAY"
echo "FLOW_RESUME_CKPT=${FLOW_RESUME_CKPT:-<none>}"

nvidia-smi
start_gpu_log "$RUN_DIR/gpu_metrics.csv"

SUMMARY_TSV="$RUN_DIR/cycle_summary.tsv"
printf "cycle\tpath_teacher_ckpt\tpath_init_ckpt\tpath_final_ckpt\tflow_init_ckpt\tflow_final_ckpt\tflow_final_ema_ckpt\tpath_experiment_dir\tflow_experiment_dir\tfid_tsv\n" > "$SUMMARY_TSV"

current_flow_ckpt="$INITIAL_FLOW_CKPT"
current_path_ckpt="$INITIAL_PATH_CKPT"

build_path_arch_args() {
  local -a args=(
    --path-arch "$PATH_ARCH"
  )
  if [[ "$PATH_ARCH" == "dual_stem_teacher_residual" || "$PATH_ARCH" == "dual_stem_teacher_residual_subboundary" || "$PATH_ARCH" == "dual_stem_direct_residual" ]]; then
    if [[ "$PATH_USE_ENDPOINT_CONDITIONING" == "1" ]]; then
      args+=(--path-use-endpoint-conditioning)
    elif [[ "$PATH_USE_ENDPOINT_CONDITIONING" == "0" ]]; then
      args+=(--no-path-use-endpoint-conditioning)
    else
      echo "PATH_USE_ENDPOINT_CONDITIONING must be 0 or 1, got '$PATH_USE_ENDPOINT_CONDITIONING'." >&2
      exit 1
    fi
  fi
  printf '%s\n' "${args[@]}"
}

find_latest_experiment_dir() {
  local root="$1"
  find "$root" -mindepth 1 -maxdepth 1 -type d | sort | tail -1
}

fid_value_for_tag() {
  local fid_tsv="$1"
  local tag="$2"
  if [[ ! -f "$fid_tsv" ]]; then
    return 0
  fi
  awk -F'\t' -v tag="$tag" '
    $1 == tag && $2 ~ /^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $2 }
    END { if (value != "") print value }
  ' "$fid_tsv"
}

start_cycle=1
if [[ -n "$RESUME_FROM_CYCLE_DIR" ]]; then
  resume_cycle_name="$(basename "$RESUME_FROM_CYCLE_DIR")"
  if [[ ! "$resume_cycle_name" =~ ^cycle_([0-9]+)$ ]]; then
    echo "RESUME_FROM_CYCLE_DIR must point to a cycle_XX directory, got '$RESUME_FROM_CYCLE_DIR'." >&2
    exit 1
  fi
  resume_path_experiment_dir="$(find_latest_experiment_dir "$RESUME_FROM_CYCLE_DIR/path_results")"
  resume_flow_experiment_dir="$(find_latest_experiment_dir "$RESUME_FROM_CYCLE_DIR/flow_results")"
  if [[ -z "${resume_path_experiment_dir:-}" || -z "${resume_flow_experiment_dir:-}" ]]; then
    echo "Could not locate path/flow experiment directories under $RESUME_FROM_CYCLE_DIR." >&2
    exit 1
  fi
  current_path_ckpt="$resume_path_experiment_dir/checkpoints/$(printf "%07d" "$PATH_STAGE_STEPS").pt"
  current_flow_ckpt="$resume_flow_experiment_dir/checkpoints/$(printf "%07d" "$FLOW_STAGE_STEPS")_ema.pt"
  start_cycle=$((10#${BASH_REMATCH[1]} + 1))
fi

for cycle in $(seq "$start_cycle" "$NUM_CYCLES"); do
  cycle_tag=$(printf "cycle_%02d" "$cycle")
  cycle_dir="$RUN_DIR/$cycle_tag"
  path_results_root="$cycle_dir/path_results"
  flow_results_root="$cycle_dir/flow_results"
  fid_out_root="$cycle_dir/fid_eval_${NUM_FID_SAMPLES}"
  mkdir -p "$cycle_dir" "$path_results_root" "$flow_results_root" "$fid_out_root"
  prev_flow_ckpt="$current_flow_ckpt"
  path_init_ckpt="$INITIAL_PATH_CKPT"
  path_subtractor_ckpt="$INITIAL_PATH_CKPT"
  path_teacher_ckpt="$prev_flow_ckpt"
  if [[ "$cycle" -eq 1 ]]; then
    path_teacher_ckpt="$PATH_TEACHER_CKPT"
  fi
  path_experiment_dir="$(find_latest_experiment_dir "$path_results_root")"
  flow_experiment_dir="$(find_latest_experiment_dir "$flow_results_root")"
  path_final_ckpt=""
  if [[ -n "${path_experiment_dir:-}" ]]; then
    path_final_ckpt="$path_experiment_dir/checkpoints/$(printf "%07d" "$PATH_STAGE_STEPS").pt"
  fi
  flow_final_ckpt=""
  flow_final_ema_ckpt=""
  flow_final_ema_tag="$(printf "%07d" "$FLOW_STAGE_STEPS")_ema"
  if [[ -n "${flow_experiment_dir:-}" ]]; then
    flow_final_ckpt="$flow_experiment_dir/checkpoints/$(printf "%07d" "$FLOW_STAGE_STEPS").pt"
    flow_final_ema_ckpt="$flow_experiment_dir/checkpoints/${flow_final_ema_tag}.pt"
  fi
  fid_tsv="$fid_out_root/fid_regression.tsv"
  fid_value="$(fid_value_for_tag "$fid_tsv" "$flow_final_ema_tag")"

  if [[ -n "${fid_value:-}" ]]; then
    if [[ -z "${path_final_ckpt:-}" || ! -f "$path_final_ckpt" ]]; then
      echo "Cycle ${cycle_tag} has FID output but missing path checkpoint." >&2
      exit 1
    fi
    if [[ -z "${flow_final_ema_ckpt:-}" || ! -f "$flow_final_ema_ckpt" ]]; then
      echo "Cycle ${cycle_tag} has FID output but missing flow EMA checkpoint." >&2
      exit 1
    fi
    echo "=== ${cycle_tag}: already complete with FID ${fid_value}; skipping cycle ==="
    current_path_ckpt="$path_final_ckpt"
    current_flow_ckpt="$flow_final_ema_ckpt"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$cycle_tag" \
      "$path_teacher_ckpt" \
      "$path_init_ckpt" \
      "$current_path_ckpt" \
      "$prev_flow_ckpt" \
      "${flow_final_ckpt:-}" \
      "$current_flow_ckpt" \
      "${path_experiment_dir:-SKIPPED}" \
      "${flow_experiment_dir:-SKIPPED}" \
      "$fid_tsv" >> "$SUMMARY_TSV"
    continue
  fi

  if [[ -n "${path_final_ckpt:-}" && -f "$path_final_ckpt" ]]; then
    echo "=== ${cycle_tag}: Stage 1 path update already complete; reusing $path_final_ckpt ==="
    current_path_ckpt="$path_final_ckpt"
  elif [[ "$cycle" -eq 1 && "$SKIP_PATH_STAGE_CYCLE1" == "1" ]]; then
    echo "=== ${cycle_tag}: Stage 1 path update skipped ==="
    echo "Reusing initial path ckpt: $path_init_ckpt"
    current_path_ckpt="$path_init_ckpt"
  else
    mapfile -t path_arch_args < <(build_path_arch_args)
    echo "=== ${cycle_tag}: Stage 1 path update ==="
    echo "Teacher flow ckpt: $path_teacher_ckpt"
    echo "Path init ckpt: $path_init_ckpt"
    echo "Path subtractor ckpt: $path_subtractor_ckpt"
    start_gpu_log "$cycle_dir/path_gpu_metrics.csv"
    "$ENV_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train_path.py \
      --model "$PATH_MODEL" \
      --teacher-model "$SIT_MODEL" \
      --attn-func "$ATTN_FUNC" \
      --amp-dtype "$AMP_DTYPE" \
      --image-size 256 \
      --data-path "$IMAGENET_TRAIN" \
      --packed-latent-view-mode "$PACKED_LATENT_VIEW_MODE" \
      --vae ema \
      --teacher-ckpt "$path_teacher_ckpt" \
      --path-subtractor-ckpt "$path_subtractor_ckpt" \
      --path-init-ckpt "$path_init_ckpt" \
      --keep-path-init-output-layer \
      "${path_arch_args[@]}" \
      --results-dir "$path_results_root" \
      --global-batch-size "$PATH_GLOBAL_BATCH_SIZE" \
      --per-gpu-batch-size "$PATH_PER_GPU_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --beta "$PATH_BETA" \
      --fd-step "$PATH_FD_STEP" \
      --accel-reg-weight "$PATH_ACCEL_REG_WEIGHT" \
      --path-loss-mode "$PATH_LOSS_MODE" \
      --learned-path-mix "$PATH_LEARNED_PATH_MIX" \
      "${path_residual_x0_time_rho_args[@]}" \
      --x0-hat-rho-scale "$X0_HAT_RHO_SCALE" \
      --lr "$PATH_LR" \
      --lr-schedule cosine \
      --min-lr "$PATH_MIN_LR" \
      --lr-warmup-steps "$PATH_LR_WARMUP_STEPS" \
      --lr-anneal-steps "$PATH_LR_ANNEAL_STEPS" \
      --weight-decay "$PATH_WEIGHT_DECAY" \
      --path_net_ema "$PATH_NET_EMA" \
      --epochs 999999 \
      --max-train-steps "$PATH_STAGE_STEPS" \
      --log-every "$PATH_LOG_EVERY" \
      --ckpt-every "$PATH_CKPT_EVERY" \
      --autosave-every "$PATH_AUTOSAVE_EVERY" \
      --eval-every "$PATH_EVAL_EVERY" \
      --eval-num-batches "$PATH_EVAL_NUM_BATCHES"
    stop_last_gpu_log

    path_experiment_dir="$(find_latest_experiment_dir "$path_results_root")"
    if [[ -z "${path_experiment_dir:-}" ]]; then
      echo "No path experiment directory found under $path_results_root" >&2
      exit 1
    fi
    current_path_ckpt="$path_experiment_dir/checkpoints/$(printf "%07d" "$PATH_STAGE_STEPS").pt"
    if [[ ! -f "$current_path_ckpt" ]]; then
      echo "Expected path checkpoint $current_path_ckpt was not produced." >&2
      exit 1
    fi
  fi

  if [[ -n "${flow_final_ckpt:-}" && -f "$flow_final_ckpt" && -f "$flow_final_ema_ckpt" ]]; then
    echo "=== ${cycle_tag}: Stage 2 flow update already complete; reusing $flow_final_ema_ckpt ==="
  else
    echo "=== ${cycle_tag}: Stage 2 flow update ==="
    if [[ -n "$FLOW_RESUME_CKPT" && "$cycle" -eq 1 ]]; then
      echo "Flow resume ckpt: $FLOW_RESUME_CKPT"
    else
      echo "Flow init ckpt: $prev_flow_ckpt"
    fi
    echo "Latest path ckpt: $current_path_ckpt"
    start_gpu_log "$cycle_dir/flow_gpu_metrics.csv"
    flow_load_args=(--init-ckpt "$prev_flow_ckpt")
    if [[ -n "$FLOW_RESUME_CKPT" && "$cycle" -eq 1 ]]; then
      flow_load_args=(--resume "$FLOW_RESUME_CKPT")
    fi
    "$ENV_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
      --model "$SIT_MODEL" \
      --attn-func "$ATTN_FUNC" \
      --amp-dtype "$AMP_DTYPE" \
      --image-size 256 \
      --data-path "$IMAGENET_TRAIN" \
      --packed-latent-view-mode "$PACKED_LATENT_VIEW_MODE" \
      --vae ema \
      "${flow_load_args[@]}" \
      --results-dir "$flow_results_root" \
      --global-batch-size "$FLOW_GLOBAL_BATCH_SIZE" \
      --per-gpu-batch-size "$FLOW_PER_GPU_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --lr "$FLOW_LR" \
      --lr-schedule none \
      --weight-decay "$FLOW_WEIGHT_DECAY" \
      --ema-decay "$FLOW_EMA_DECAY" \
      --path-type Linear \
      --prediction velocity \
      --loss-weight None \
      --epochs 999999 \
      --max-train-steps "$FLOW_STAGE_STEPS" \
      --log-every "$FLOW_LOG_EVERY" \
      --sample-every 0 \
      --ckpt-every "$FLOW_CKPT_EVERY" \
      --autosave-every "$FLOW_AUTOSAVE_EVERY" \
      --learned-path-ckpt "$current_path_ckpt" \
      --learned-path-fd-step "$PATH_FD_STEP" \
      --learned-path-mix "$FLOW_LEARNED_PATH_MIX" \
      "${path_residual_x0_time_rho_args[@]}" \
      --x0-hat-rho-scale "$X0_HAT_RHO_SCALE" \
      --learned-path-subtractor-residual-scale "$FLOW_LEARNED_PATH_SUBTRACTOR_RESIDUAL_SCALE"
    stop_last_gpu_log

    flow_experiment_dir="$(find_latest_experiment_dir "$flow_results_root")"
    if [[ -z "${flow_experiment_dir:-}" ]]; then
      echo "No flow experiment directory found under $flow_results_root" >&2
      exit 1
    fi
    flow_final_ckpt="$flow_experiment_dir/checkpoints/$(printf "%07d" "$FLOW_STAGE_STEPS").pt"
    flow_final_ema_ckpt="$flow_experiment_dir/checkpoints/${flow_final_ema_tag}.pt"
    if [[ ! -f "$flow_final_ckpt" ]]; then
      echo "Expected flow checkpoint $flow_final_ckpt was not produced." >&2
      exit 1
    fi
    if [[ ! -f "$flow_final_ema_ckpt" ]]; then
      echo "Expected flow EMA checkpoint $flow_final_ema_ckpt was not produced." >&2
      exit 1
    fi
  fi

  echo "=== ${cycle_tag}: Stage 3 FID eval ==="
  start_gpu_log "$cycle_dir/fid_gpu_metrics.csv"
  ENV_PYTHON="$ENV_PYTHON" \
  MODEL="$SIT_MODEL" \
  ATTN_FUNC="" \
  BASELINE_FID="$BASELINE_FID" \
  final_scripts/eval_continue_fid_series.sh \
    "$flow_experiment_dir" \
    "$REF_DIR" \
    "$NUM_FID_SAMPLES" \
    "$fid_out_root" \
    "$NPROC_PER_NODE" \
    "$GEN_CONFIG_PATH"
  stop_last_gpu_log

  fid_tsv="$fid_out_root/fid_regression.tsv"
  fid_value="$(fid_value_for_tag "$fid_tsv" "$flow_final_ema_tag")"
  if [[ -z "${fid_value:-}" ]]; then
    echo "Expected numeric FID entry for ${flow_final_ema_tag} in $fid_tsv" >&2
    exit 1
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$cycle_tag" \
    "$path_teacher_ckpt" \
    "$path_init_ckpt" \
    "$current_path_ckpt" \
    "$prev_flow_ckpt" \
    "$flow_final_ckpt" \
    "$flow_final_ema_ckpt" \
    "$path_experiment_dir" \
    "$flow_experiment_dir" \
    "$fid_tsv" >> "$SUMMARY_TSV"

  current_flow_ckpt="$flow_final_ema_ckpt"
done

echo "Summary TSV: $SUMMARY_TSV"
echo "End time: $(date)"
