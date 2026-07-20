# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from contextlib import nullcontext
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import (
    SiT_models,
    build_dual_stem_path_sit,
    is_dual_stem_path_state_dict,
)
from model_guidance import (
    apply_model_guidance_correction,
    model_guidance_reference_labels,
    prepare_model_guidance_batch,
    validate_model_guidance_config,
)
from sit_utils.train_utils import (
    get_optimizer_lr, make_lr_schedule, maybe_autocast, parse_transport_args, set_optimizer_lr,
)
from path_energy import (
    disable_label_dropout,
    freeze_model_params,
    generalized_mixed_path_derivative_fd,
    sample_path_residual_x0_hat,
)
from sit_utils.latent_dataset import LATENT_SCALE, PackedVAELabelDataset, PairedPackedLatentSampler, is_packed_vae_latent_dir, paired_packed_latent_original_count, resolve_packed_latent_view_mode, sample_packed_vae_latents
from transport import create_transport, Sampler
from transport.utils import mean_flat
from diffusers.models import AutoencoderKL
from sit_utils import wandb_utils


AUTORESUME_FILENAME = "latest_autoresume.pt"
TEST_AUTOSAVE_EXIT_CODE = 85
IMAGENET_TRAIN_IMAGE_COUNT = 1_281_167
DUAL_STEM_PATH_PARAMETERIZATIONS = {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary", "dual_stem_direct_residual"}
DUAL_STEM_SUBTRACTOR_PARAMETERIZATIONS = {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary"}


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def list_experiment_dirs(results_dir):
    return sorted(
        path for path in glob(os.path.join(results_dir, "*"))
        if os.path.isdir(path)
    )


def latest_experiment_dir(results_dir):
    experiment_dirs = list_experiment_dirs(results_dir)
    return experiment_dirs[-1] if experiment_dirs else None


def resolve_experiment_dir(results_dir, experiment_name):
    existing_dirs = list_experiment_dirs(results_dir)
    if existing_dirs:
        return existing_dirs[-1], False
    return os.path.join(results_dir, experiment_name), True


def atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_torch_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def autoresume_checkpoint_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, AUTORESUME_FILENAME)


def normalize_ema_decay_label(decay):
    decay = float(decay)
    if decay == 0.0:
        decay = 0.0
    return str(decay)


def ema_checkpoint_key(decay):
    return f"ema{normalize_ema_decay_label(decay)}"


def parse_saved_ema_list(value):
    if value is None:
        return None
    saved_ema_list = []
    seen_labels = set()
    for raw_decay in value.split(","):
        raw_decay = raw_decay.strip()
        if not raw_decay:
            raise argparse.ArgumentTypeError("--saved-ema-list cannot contain empty values.")
        try:
            decay = float(raw_decay)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--saved-ema-list contains a non-float value: {raw_decay}"
            ) from exc
        if not np.isfinite(decay) or decay < 0.0 or decay > 1.0:
            raise argparse.ArgumentTypeError("--saved-ema-list values must be in [0, 1].")
        label = normalize_ema_decay_label(decay)
        if label in seen_labels:
            raise argparse.ArgumentTypeError(
                f"--saved-ema-list contains duplicate EMA decay {label}."
            )
        seen_labels.add(label)
        saved_ema_list.append((label, decay))
    if not saved_ema_list:
        raise argparse.ArgumentTypeError("--saved-ema-list must contain at least one value.")
    return saved_ema_list


def checkpoint_multi_ema_keys(checkpoint):
    if not isinstance(checkpoint, dict):
        return []
    return sorted(key for key in checkpoint if key.startswith("ema") and key != "ema")


def load_resume_ema_state(resume_obj, ema, ema_models, args):
    if args.saved_ema_list is None:
        if "ema" in resume_obj:
            ema.load_state_dict(resume_obj["ema"], strict=True)
            return
        ema_key = ema_checkpoint_key(args.ema_decay)
        if ema_key not in resume_obj:
            available = ", ".join(checkpoint_multi_ema_keys(resume_obj)) or "none"
            raise ValueError(
                f"Resume checkpoint does not contain legacy 'ema' or matching '{ema_key}'. "
                f"Available multi-EMA keys: {available}."
            )
        ema.load_state_dict(resume_obj[ema_key], strict=True)
        return

    if "ema" in resume_obj:
        raise ValueError(
            "Cannot resume with --saved-ema-list from a legacy checkpoint containing only 'ema'. "
            "Resume from a checkpoint saved with the same multi-EMA format."
        )
    missing = [key for key in ema_models if key not in resume_obj]
    if missing:
        available = ", ".join(checkpoint_multi_ema_keys(resume_obj)) or "none"
        raise ValueError(
            f"Resume checkpoint is missing requested EMA keys: {missing}. "
            f"Available multi-EMA keys: {available}."
        )
    for key, ema_model in ema_models.items():
        ema_model.load_state_dict(resume_obj[key], strict=True)


def fast_forward_loader(loader_iter, num_micro_batches, *, logger, epoch, train_steps):
    if num_micro_batches <= 0:
        return
    logger.info(
        f"Fast-forwarding {num_micro_batches} micro-batches for epoch {epoch} "
        f"to resume from optimizer step {train_steps}."
    )
    for _ in range(num_micro_batches):
        try:
            next(loader_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "Could not fast-forward the dataloader to the requested resume position."
            ) from exc


def maybe_exit_after_autosave(train_steps, total_train_steps, logger):
    if os.environ.get("SIT_TEST_EXIT_AFTER_AUTOSAVE") != "1":
        return
    if train_steps >= total_train_steps:
        return
    logger.info(f"Exiting after autosave at optimizer step {train_steps} for test coverage.")
    cleanup()
    raise SystemExit(TEST_AUTOSAVE_EXIT_CODE)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def unwrap_checkpoint_for_weights(obj):
    if isinstance(obj, dict):
        if "ema" in obj:
            return obj["ema"]
        if "model" in obj:
            return obj["model"]
    return obj


def unwrap_path_checkpoint_for_weights(obj):
    if isinstance(obj, dict) and "ema" in obj:
        return obj["ema"]
    return unwrap_checkpoint_for_weights(obj)


def infer_learn_sigma_from_state_dict(state_dict, model_name):
    patch_size = int(model_name.split("/")[-1].split("-")[0])
    out_dim = state_dict["final_layer.linear.weight"].shape[0]
    no_sigma_dim = (patch_size ** 2) * 4
    sigma_dim = (patch_size ** 2) * 8
    if out_dim == sigma_dim:
        return True
    if out_dim == no_sigma_dim:
        return False
    raise ValueError(
        f"Could not infer learn_sigma from final head shape {out_dim} for model {model_name}"
    )


def normalize_path_parameterization(path_arch):
    if path_arch in (None, "legacy", "legacy_sit"):
        return "legacy"
    if path_arch in DUAL_STEM_PATH_PARAMETERIZATIONS:
        return path_arch
    raise ValueError(f"Unsupported path parameterization: {path_arch}")


def infer_dual_stem_endpoint_conditioning(state_dict):
    has_endpoint_keys = any(key.startswith("endpoint_conditioner.") for key in state_dict)
    if has_endpoint_keys:
        required_keys = {
            "endpoint_conditioner.proj.weight",
            "endpoint_conditioner.proj.bias",
        }
        missing_keys = sorted(key for key in required_keys if key not in state_dict)
        if missing_keys:
            raise ValueError(
                "Dual-stem path checkpoint has a partial endpoint-conditioning branch; "
                f"missing keys: {missing_keys}"
            )
    return has_endpoint_keys


def infer_path_parameterization(path_checkpoint_obj, path_state_dict):
    checkpoint_args = path_checkpoint_obj.get("args", {}) if isinstance(path_checkpoint_obj, dict) else {}
    path_arch = checkpoint_args.get("path_arch")
    if path_arch is not None:
        normalized = normalize_path_parameterization(path_arch)
        if normalized == "legacy" and is_dual_stem_path_state_dict(path_state_dict):
            raise ValueError("Checkpoint args declare a legacy path-net, but the weights are dual-stem.")
        if normalized in DUAL_STEM_PATH_PARAMETERIZATIONS and not is_dual_stem_path_state_dict(path_state_dict):
            raise ValueError("Checkpoint args declare a dual-stem path-net, but the weights look like a legacy SiT.")
        return normalized
    if is_dual_stem_path_state_dict(path_state_dict):
        return "dual_stem_teacher_residual"
    return "legacy"


def load_frozen_learned_path_components(path_ckpt, *, model_name, latent_size, num_classes, device, attn_func=None):
    path_checkpoint_obj = load_torch_checkpoint(path_ckpt)
    path_state_dict = unwrap_path_checkpoint_for_weights(path_checkpoint_obj)
    path_parameterization = infer_path_parameterization(path_checkpoint_obj, path_state_dict)
    checkpoint_args = path_checkpoint_obj.get("args", {}) if isinstance(path_checkpoint_obj, dict) else {}
    path_model_name = checkpoint_args.get("model") or model_name
    teacher_model_name = checkpoint_args.get("teacher_model") or path_model_name
    path_learn_sigma = infer_learn_sigma_from_state_dict(path_state_dict, path_model_name)

    if path_parameterization == "legacy":
        path_model = SiT_models[path_model_name](input_size=latent_size, num_classes=num_classes, learn_sigma=path_learn_sigma, attn_func=attn_func).to(device)
        path_model.load_state_dict(path_state_dict, strict=True)
        path_energy_teacher = None
        path_subtractor = None
        metadata = {
            "path_parameterization": path_parameterization,
            "path_model_name": path_model_name,
            "teacher_model_name": None,
            "teacher_ckpt": None,
            "path_subtractor_ckpt": None,
            "path_boundary_envelope_lambda": None,
            "path_use_endpoint_conditioning": None,
        }
    elif path_parameterization in DUAL_STEM_PATH_PARAMETERIZATIONS:
        use_endpoint_conditioning = infer_dual_stem_endpoint_conditioning(path_state_dict)
        path_boundary_envelope_lambda = checkpoint_args.get("path_boundary_envelope_lambda", 1.0)
        path_model = build_dual_stem_path_sit(path_model_name, input_size=latent_size, num_classes=num_classes, learn_sigma=path_learn_sigma, use_endpoint_conditioning=use_endpoint_conditioning, teacher_residual_boundary_lambda=path_boundary_envelope_lambda, attn_func=attn_func).to(device)
        path_model.load_state_dict(path_state_dict, strict=True)
        teacher_ckpt = checkpoint_args.get("teacher_ckpt")
        path_subtractor_ckpt = checkpoint_args.get("path_subtractor_ckpt")

        if path_parameterization == "dual_stem_direct_residual":
            path_energy_teacher = None
            path_subtractor = None
        else:
            if not teacher_ckpt:
                raise ValueError("Dual-stem learned-path checkpoints require args['teacher_ckpt'] so downstream flow training can reconstruct the teacher-residualized path.")
            path_subtractor_ckpt = path_subtractor_ckpt or teacher_ckpt

            energy_teacher_state_dict = unwrap_checkpoint_for_weights(load_torch_checkpoint(teacher_ckpt))
            energy_teacher_learn_sigma = infer_learn_sigma_from_state_dict(energy_teacher_state_dict, teacher_model_name)
            if energy_teacher_learn_sigma != path_learn_sigma:
                raise ValueError("Dual-stem learned path and its saved teacher checkpoint disagree on learn_sigma.")
            path_energy_teacher = SiT_models[teacher_model_name](input_size=latent_size, num_classes=num_classes, learn_sigma=energy_teacher_learn_sigma, attn_func=attn_func).to(device)
            path_energy_teacher.load_state_dict(energy_teacher_state_dict, strict=True)
            disable_label_dropout(path_energy_teacher)
            freeze_model_params(path_energy_teacher)
            path_energy_teacher.eval()

            if os.path.abspath(path_subtractor_ckpt) == os.path.abspath(teacher_ckpt) and path_model_name == teacher_model_name:
                path_subtractor = path_energy_teacher
            else:
                path_subtractor_state_dict = unwrap_checkpoint_for_weights(load_torch_checkpoint(path_subtractor_ckpt))
                path_subtractor_learn_sigma = infer_learn_sigma_from_state_dict(path_subtractor_state_dict, path_model_name)
                if path_subtractor_learn_sigma != path_learn_sigma:
                    raise ValueError("Dual-stem learned path and its saved path-subtractor checkpoint disagree on learn_sigma.")
                path_subtractor = SiT_models[path_model_name](input_size=latent_size, num_classes=num_classes, learn_sigma=path_subtractor_learn_sigma, attn_func=attn_func).to(device)
                path_subtractor.load_state_dict(path_subtractor_state_dict, strict=True)
                disable_label_dropout(path_subtractor)
                freeze_model_params(path_subtractor)
                path_subtractor.eval()

        metadata = {
            "path_parameterization": path_parameterization,
            "path_model_name": path_model_name,
            "teacher_model_name": teacher_model_name,
            "teacher_ckpt": teacher_ckpt,
            "path_subtractor_ckpt": path_subtractor_ckpt,
            "path_boundary_envelope_lambda": path_boundary_envelope_lambda,
            "path_use_endpoint_conditioning": use_endpoint_conditioning,
        }
    else:
        raise ValueError(f"Unsupported path parameterization: {path_parameterization}")

    disable_label_dropout(path_model)
    freeze_model_params(path_model)
    path_model.eval()
    return path_model, path_energy_teacher, path_subtractor, metadata


def create_learned_path_plan(
    path_model,
    *,
    fd_step,
    learned_mix,
    energy_teacher=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    path_parameterization=None,
    disable_path_residual_x0_time_rho=False,
    path_rho_constant=None,
    x0_hat_rho_scale=1.0,
):
    def plan_fn(*, t, x0, x1, model_kwargs):
        y = model_kwargs.get("y")
        if y is None:
            raise ValueError("A learned-path training plan requires class labels in model_kwargs['y'].")
        with torch.no_grad():
            x0_hat = sample_path_residual_x0_hat(
                x0,
                t,
                disable_path_residual_x0_time_rho=disable_path_residual_x0_time_rho,
                path_rho_constant=path_rho_constant,
                x0_hat_rho_scale=x0_hat_rho_scale,
            )
            xt, ut = generalized_mixed_path_derivative_fd(
                path_model,
                energy_teacher,
                x0,
                x1,
                y,
                t,
                h=fd_step,
                learned_mix=learned_mix,
                path_parameterization=path_parameterization,
                path_subtractor=path_subtractor,
                subtractor_residual_scale=subtractor_residual_scale,
                x0_hat=x0_hat,
            )
        return t, xt, ut

    return plan_fn


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    if args.per_gpu_batch_size is not None:
        local_batch_size = args.per_gpu_batch_size
        global_micro_batch_size = local_batch_size * world_size
        if args.global_batch_size % global_micro_batch_size != 0:
            raise ValueError(
                "--global-batch-size must be divisible by WORLD_SIZE * --per-gpu-batch-size."
            )
        args.grad_accum_steps = args.global_batch_size // global_micro_batch_size
        global_batch_size = args.global_batch_size
    else:
        assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
        local_batch_size = int(args.global_batch_size // world_size)
        global_micro_batch_size = local_batch_size * world_size
        global_batch_size = global_micro_batch_size * args.grad_accum_steps
    args.per_gpu_batch_size = local_batch_size
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    print(
        f"Starting rank={rank}, seed={seed}, world_size={world_size}, "
        f"per_gpu_batch_size={local_batch_size}, grad_accum_steps={args.grad_accum_steps}."
    )

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        proposed_experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        )
        experiment_dir, is_new_experiment = resolve_experiment_dir(
            args.results_dir,
            proposed_experiment_name,
        )
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        experiment_name = os.path.basename(experiment_dir)
        logger = create_logger(experiment_dir)
        if is_new_experiment:
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            logger.info(f"Reusing experiment directory at {experiment_dir}")

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    dist.barrier()
    if rank != 0:
        experiment_dir = latest_experiment_dir(args.results_dir)
        if experiment_dir is None:
            raise RuntimeError(f"No experiment directory found under {args.results_dir}")
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        attn_func=args.attn_func,
        class_dropout_prob=args.class_dropout_prob,
        learn_sigma=args.learn_sigma,
        always_allocate_uncond_slot=(args.mg_start_step >= 0),
    )

    if args.mg_start_step >= 0:
        if args.class_dropout_prob != 0.0:
            logger.warning(
                "MG is enabled with random class dropout. The reference MG recipe uses "
                "--class-dropout-prob 0.0 to avoid compounding dropout mechanisms."
            )
        if args.learn_sigma:
            logger.warning(
                "MG is enabled with learned sigma. The reference MG recipe uses --no-learn-sigma."
            )

    # Note that parameter initialization is done within the SiT constructor
    ema_models = OrderedDict()
    ema_decays = OrderedDict()
    if args.saved_ema_list is None:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        ema_models["ema"] = ema
        ema_decays["ema"] = args.ema_decay
    else:
        for ema_label, ema_decay in args.saved_ema_list:
            ema_key = f"ema{ema_label}"
            ema_model = deepcopy(model).to(device)
            requires_grad(ema_model, False)
            ema_models[ema_key] = ema_model
            ema_decays[ema_key] = ema_decay
        ema = next(iter(ema_models.values()))  # Used for periodic sampling.

    model = DDP(model.to(device), device_ids=[device])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    start_steps_in_epoch = 0
    train_steps = 0
    resumed_from_checkpoint = False
    resume_path = None
    resume_source = None

    if args.auto_resume:
        candidate_autoresume_path = autoresume_checkpoint_path(checkpoint_dir)
        if os.path.isfile(candidate_autoresume_path):
            resume_path = candidate_autoresume_path
            resume_source = "latest_autoresume.pt"
    if resume_path is None and args.resume is not None:
        resume_path = args.resume
        resume_source = "--resume"

    init_ckpt_path = args.init_ckpt or args.ckpt
    if init_ckpt_path is not None:
        init_obj = load_torch_checkpoint(init_ckpt_path)
        init_state = unwrap_checkpoint_for_weights(init_obj)
        model.module.load_state_dict(init_state, strict=True)
        for ema_model in ema_models.values():
            ema_model.load_state_dict(init_state, strict=True)

    if resume_path is not None:
        logger.info(f"Loading training state from {resume_path} ({resume_source})")
        resume_obj = load_torch_checkpoint(resume_path)
        model.module.load_state_dict(resume_obj["model"], strict=True)
        load_resume_ema_state(resume_obj, ema, ema_models, args)
        if not args.reset_opt_on_resume:
            opt.load_state_dict(resume_obj["opt"])
            if "scaler" in resume_obj and scaler.is_enabled():
                scaler.load_state_dict(resume_obj["scaler"])
        start_epoch = resume_obj.get("epoch", 0)
        start_steps_in_epoch = resume_obj.get("steps_in_epoch", 0)
        train_steps = resume_obj.get("train_steps", 0)
        resumed_from_checkpoint = True
    set_optimizer_lr(opt, args.lr)

    learned_path_model = None
    learned_path_energy_teacher = None
    learned_path_subtractor = None
    learned_path_metadata = None
    training_plan = None
    if args.learned_path_ckpt is not None:
        if args.prediction != "velocity":
            raise ValueError("Learned-path flow matching currently supports only --prediction velocity.")
        (
            learned_path_model,
            learned_path_energy_teacher,
            learned_path_subtractor,
            learned_path_metadata,
        ) = load_frozen_learned_path_components(
            args.learned_path_ckpt,
            model_name=args.model,
            latent_size=latent_size,
            num_classes=args.num_classes,
            device=device,
            attn_func=args.attn_func,
        )
        training_plan = create_learned_path_plan(
            learned_path_model,
            fd_step=args.learned_path_fd_step,
            learned_mix=args.learned_path_mix,
            energy_teacher=learned_path_energy_teacher,
            path_subtractor=learned_path_subtractor,
            subtractor_residual_scale=args.learned_path_subtractor_residual_scale,
            path_parameterization=learned_path_metadata["path_parameterization"],
            disable_path_residual_x0_time_rho=args.disable_path_residual_x0_time_rho,
            path_rho_constant=args.path_rho_constant,
            x0_hat_rho_scale=args.x0_hat_rho_scale,
        )

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    use_packed_latents = is_packed_vae_latent_dir(args.data_path)
    packed_latent_view_mode = "all"
    vae = None
    if (not use_packed_latents) or args.sample_every > 0:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data:
    if use_packed_latents:
        dataset = PackedVAELabelDataset(args.data_path)
        packed_latent_view_mode = resolve_packed_latent_view_mode(args.packed_latent_view_mode, len(dataset), expected_original_count=IMAGENET_TRAIN_IMAGE_COUNT, tolerance=global_batch_size)
        logger.info(
            f"Using packed VAE latents from {args.data_path}; "
            "sampling final 4-channel latents from stored mean/std at runtime."
        )
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        logger.info(
            f"Using image dataset from {args.data_path}; encoding images to latents with the VAE at runtime."
        )
    if use_packed_latents and packed_latent_view_mode == "one-per-image":
        sampler = PairedPackedLatentSampler(num_originals=paired_packed_latent_original_count(len(dataset)), num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed, view_policy="random")
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if use_packed_latents and packed_latent_view_mode == "one-per-image":
        effective_num_images = paired_packed_latent_original_count(len(dataset))
        logger.info(
            f"Dataset contains {len(dataset):,} packed latent views ({args.data_path}); "
            f"packed-latent view mode resolved to one-per-image from --packed-latent-view-mode={args.packed_latent_view_mode}, "
            f"so each epoch samples one stored view for each of {effective_num_images:,} original images."
        )
    else:
        dataset_units = "packed latent samples" if use_packed_latents else "images"
        logger.info(f"Dataset contains {len(dataset):,} {dataset_units} ({args.data_path})")
    logger.info(
        f"Using per_gpu_batch_size={local_batch_size}, "
        f"global_batch_size={global_batch_size}, "
        f"grad_accum_steps={args.grad_accum_steps}"
    )
    logger.info(f"Attention backend: {args.attn_func or 'default'}, autocast: {args.amp_dtype}")
    if args.saved_ema_list is None:
        logger.info(f"EMA decay: {args.ema_decay:.6f}")
    else:
        logger.info(
            "Saved EMA decays: "
            f"{', '.join(label for label, _ in args.saved_ema_list)} "
            "(--ema-decay ignored for multi-EMA tracking)"
        )

    steps_per_epoch = len(loader) // args.grad_accum_steps
    if steps_per_epoch < 1:
        raise ValueError(
            "Not enough batches per epoch for the requested global/per-GPU batch configuration."
        )
    while start_steps_in_epoch >= steps_per_epoch:
        start_steps_in_epoch -= steps_per_epoch
        start_epoch += 1
    total_train_steps = args.max_train_steps if args.max_train_steps is not None else args.epochs * steps_per_epoch
    lr_anneal_steps = args.lr_anneal_steps
    if args.lr_schedule != "none":
        if lr_anneal_steps is None:
            lr_anneal_steps = max(total_train_steps - train_steps, 1)
        lr_schedule = make_lr_schedule(
            schedule=args.lr_schedule,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.lr_warmup_steps,
            anneal_steps=lr_anneal_steps,
            start_step=train_steps,
        )
        set_optimizer_lr(opt, lr_schedule(train_steps))
    else:
        lr_schedule = None
    if learned_path_model is not None:
        learned_path_log = (
            f"Using learned path checkpoint {args.learned_path_ckpt} "
            f"with learned_path_mix={args.learned_path_mix:.3f}, "
            f"fd_step={args.learned_path_fd_step:.3f}"
        )
        if learned_path_metadata["path_parameterization"] in DUAL_STEM_SUBTRACTOR_PARAMETERIZATIONS and args.learned_path_subtractor_residual_scale != 1.0:
            learned_path_log += f", subtractor_residual_scale={args.learned_path_subtractor_residual_scale:.6f}"
        logger.info(learned_path_log)
        logger.info(f"Learned path model family: {learned_path_metadata['path_model_name']}")
        if learned_path_metadata["teacher_model_name"] is not None:
            logger.info(f"Learned path teacher family: {learned_path_metadata['teacher_model_name']}")
        logger.info(f"Learned path parameterization: {learned_path_metadata['path_parameterization']}")
        if learned_path_metadata["path_use_endpoint_conditioning"] is not None:
            endpoint_status = "enabled" if learned_path_metadata["path_use_endpoint_conditioning"] else "disabled"
            logger.info(f"Dual-stem endpoint conditioning: {endpoint_status}")
        if (
            learned_path_metadata["path_parameterization"] in {"dual_stem_teacher_residual", "dual_stem_direct_residual"}
            and learned_path_metadata["path_boundary_envelope_lambda"] is not None
        ):
            logger.info(f"Dual-stem boundary envelope lambda: {learned_path_metadata['path_boundary_envelope_lambda']:.6f}")
        if learned_path_metadata["path_parameterization"] == "dual_stem_direct_residual":
            logger.info("Learned path teacher/subtractor: ignored for dual_stem_direct_residual.")
            if args.learned_path_subtractor_residual_scale != 1.0:
                logger.info(
                    "Ignoring --learned-path-subtractor-residual-scale for dual_stem_direct_residual because no subtractor is used."
                )
            if learned_path_metadata["teacher_ckpt"] is not None:
                logger.info(f"Ignoring saved learned path teacher checkpoint for dual_stem_direct_residual: {learned_path_metadata['teacher_ckpt']}")
            if learned_path_metadata["path_subtractor_ckpt"] is not None:
                logger.info(f"Ignoring saved learned path subtractor checkpoint for dual_stem_direct_residual: {learned_path_metadata['path_subtractor_ckpt']}")
        elif learned_path_metadata["teacher_ckpt"] is not None:
            logger.info(f"Learned path energy-teacher checkpoint: {learned_path_metadata['teacher_ckpt']}")
        if (
            learned_path_metadata["path_parameterization"] in DUAL_STEM_SUBTRACTOR_PARAMETERIZATIONS
            and learned_path_metadata["path_subtractor_ckpt"] is not None
        ):
            logger.info(f"Learned path subtractor checkpoint: {learned_path_metadata['path_subtractor_ckpt']}")

    if resumed_from_checkpoint:
        logger.info(
            f"Resuming from epoch={start_epoch}, steps_in_epoch={start_steps_in_epoch}, "
            f"train_steps={train_steps}"
        )
    if train_steps >= total_train_steps:
        logger.info(
            f"Saved train_steps={train_steps} already reaches/exceeds total_train_steps={total_train_steps}; "
            "nothing to do."
        )
        cleanup()
        return

    # Prepare models for training:
    if not resumed_from_checkpoint:
        for ema_model in ema_models.values():
            update_ema(ema_model, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    for ema_model in ema_models.values():
        ema_model.eval()  # EMA models should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    resume_steps_in_epoch = start_steps_in_epoch
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_steps_completed = resume_steps_in_epoch if epoch == start_epoch else 0
        logger.info(
            f"Beginning epoch {epoch}..."
            if epoch_steps_completed == 0
            else f"Beginning epoch {epoch} at resumed optimizer-step offset {epoch_steps_completed}..."
        )
        loader_iter = iter(loader)
        fast_forward_loader(
            loader_iter,
            epoch_steps_completed * args.grad_accum_steps,
            logger=logger,
            epoch=epoch,
            train_steps=train_steps,
        )
        resume_steps_in_epoch = 0
        while epoch_steps_completed < steps_per_epoch:
            if lr_schedule is not None:
                set_optimizer_lr(opt, lr_schedule(train_steps))
            opt.zero_grad()
            accum_loss = 0.0
            accum_micro_steps = 0

            for accum_idx in range(args.grad_accum_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    break

                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    # Map input images to latent space, or sample final latents
                    # from stored VAE mean/std statistics.
                    with maybe_autocast("cuda", args.amp_dtype):
                        if use_packed_latents:
                            x = sample_packed_vae_latents(x)
                        else:
                            x = vae.encode(x).latent_dist.sample().mul_(LATENT_SCALE)
                sync_context = (
                    model.no_sync()
                    if args.grad_accum_steps > 1 and accum_idx < args.grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with maybe_autocast("cuda", args.amp_dtype):
                        mg_configured = args.mg_start_step >= 0
                        mg_active = mg_configured and train_steps >= args.mg_start_step
                        if mg_configured:
                            y_for_model, mg_weights, num_guided = prepare_model_guidance_batch(
                                y,
                                num_classes=args.num_classes,
                                weight_low=args.mgw[0],
                                weight_high=args.mgw[1],
                                drop_fraction=args.mg_data_ratio[1],
                                guidance_active=mg_active,
                                dtype=torch.float32,
                            )
                        else:
                            y_for_model, mg_weights, num_guided = y, None, 0
                        model_kwargs = dict(y=y_for_model)
                        loss_dict = transport.training_losses(model, x, model_kwargs, plan_fn=training_plan)
                        if mg_active and num_guided > 0:
                            with torch.no_grad():
                                reference_y = model_guidance_reference_labels(
                                    y[:num_guided],
                                    num_classes=args.num_classes,
                                    contrastive=args.mg_contrastive,
                                )
                                conditional_prediction = ema(
                                    loss_dict["xt"][:num_guided],
                                    loss_dict["t"][:num_guided],
                                    y[:num_guided],
                                )
                                reference_prediction = ema(
                                    loss_dict["xt"][:num_guided],
                                    loss_dict["t"][:num_guided],
                                    reference_y,
                                )
                                augmented_target = apply_model_guidance_correction(
                                    loss_dict["ut"],
                                    conditional_prediction,
                                    reference_prediction,
                                    loss_dict["t"],
                                    mg_weights,
                                    data_side_threshold=args.mg_data_side_threshold,
                                )
                            loss_dict["loss"] = mean_flat(
                                (loss_dict["model_output"] - augmented_target) ** 2
                            )
                        loss = loss_dict["loss"].mean()
                    scaled_loss = scaler.scale(loss / args.grad_accum_steps) if scaler.is_enabled() else (loss / args.grad_accum_steps)
                    scaled_loss.backward()
                accum_loss += loss.item()
                accum_micro_steps += 1

            if accum_micro_steps == 0:
                break
            if accum_micro_steps < args.grad_accum_steps:
                opt.zero_grad()
                break

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if args.saved_ema_list is None:
                update_ema(ema, model.module, decay=args.ema_decay)
            else:
                for ema_key, ema_model in ema_models.items():
                    update_ema(ema_model, model.module, decay=ema_decays[ema_key])

            # Log loss values:
            running_loss += accum_loss / args.grad_accum_steps
            log_steps += 1
            train_steps += 1
            epoch_steps_completed += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                current_lr = get_optimizer_lr(opt)
                images_per_sec = steps_per_sec * global_batch_size

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"lr={current_lr:.8f}, "
                    f"per_gpu_batch_size={local_batch_size}, "
                    f"grad_accum_steps={args.grad_accum_steps}, "
                    f"global_batch_size={global_batch_size}, "
                    f"peak_cuda_allocated_mib={peak_alloc_mib:.1f}, "
                    f"peak_cuda_reserved_mib={peak_reserved_mib:.1f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"Train Images/Sec: {images_per_sec:.2f}"
                )
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss,
                            "train lr": current_lr,
                            "train per_gpu_batch_size": local_batch_size,
                            "train grad_accum_steps": args.grad_accum_steps,
                            "train global_batch_size": global_batch_size,
                            "train peak_cuda_allocated_mib": peak_alloc_mib,
                            "train peak_cuda_reserved_mib": peak_reserved_mib,
                            "train steps/sec": steps_per_sec,
                            "train images/sec": images_per_sec,
                        },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            is_final_step = train_steps >= total_train_steps
            should_autosave = (
                args.autosave_every > 0
                and train_steps > 0
                and train_steps % args.autosave_every == 0
            )
            if should_autosave or is_final_step:
                if rank == 0:
                    autoresume_checkpoint = {
                        "model": model.module.state_dict(),
                    }
                    if args.saved_ema_list is None:
                        autoresume_checkpoint["ema"] = ema.state_dict()
                    else:
                        for ema_key, ema_model in ema_models.items():
                            autoresume_checkpoint[ema_key] = ema_model.state_dict()
                    autoresume_checkpoint.update({
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "steps_in_epoch": epoch_steps_completed,
                        "train_steps": train_steps,
                    })
                    if scaler.is_enabled():
                        autoresume_checkpoint["scaler"] = scaler.state_dict()
                    autoresume_path = autoresume_checkpoint_path(checkpoint_dir)
                    atomic_torch_save(autoresume_checkpoint, autoresume_path)
                    logger.info(f"Saved autoresume checkpoint to {autoresume_path}")
                dist.barrier()
                maybe_exit_after_autosave(train_steps, total_train_steps, logger)

            # Save SiT checkpoint:
            if args.ckpt_every > 0 and train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                    }
                    if args.saved_ema_list is None:
                        checkpoint["ema"] = ema.state_dict()
                    else:
                        for ema_key, ema_model in ema_models.items():
                            checkpoint[ema_key] = ema_model.state_dict()
                    checkpoint.update({
                        "opt": opt.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "steps_in_epoch": epoch_steps_completed,
                        "train_steps": train_steps,
                    })
                    if scaler.is_enabled():
                        checkpoint["scaler"] = scaler.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if args.saved_ema_list is None:
                        ema_only_path = f"{checkpoint_dir}/{train_steps:07d}_ema.pt"
                        torch.save(ema.state_dict(), ema_only_path)
                    else:
                        for ema_key, ema_model in ema_models.items():
                            ema_only_path = f"{checkpoint_dir}/{train_steps:07d}_{ema_key}.pt"
                            torch.save(ema_model.state_dict(), ema_only_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)
                    samples = vae.decode(samples / LATENT_SCALE).sample
                    out_samples = torch.zeros((global_micro_batch_size, 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

            if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                break

        if args.max_train_steps is not None and train_steps >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    logger.info(
        f"Peak CUDA memory: allocated_mib={peak_alloc_mib:.1f}, reserved_mib={peak_reserved_mib:.1f}"
    )
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Image dataset root or packed latent directory containing latents.bin and meta.npz.",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--per-gpu-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Number of micro-batches to accumulate per optimizer step.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--packed-latent-view-mode", type=str, default="auto", choices=["auto", "all", "one-per-image"], help="Sampling semantics for packed latent datasets. 'one-per-image' samples exactly one of the two pre-encoded views per source image each epoch; 'all' keeps the historical 2N-record behavior; 'auto' enables one-per-image semantics for the doubled full-ImageNet packed format from preproccessing/README.md.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--autosave-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Initialize model/EMA from a pretrained checkpoint or saved EMA state_dict, but start a fresh optimizer.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full training checkpoint saved by this script.")
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from checkpoints/latest_autoresume.pt in the reused experiment directory when present.",
    )
    parser.add_argument("--reset-opt-on-resume", action="store_true",
                        help="Load model weights from --resume but start a fresh optimizer state.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-anneal-steps", type=int, default=None,
                        help="Number of steps over which to anneal the LR starting from the current train step.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--saved-ema-list", type=parse_saved_ema_list, default=None,
                        help="Optional comma-separated EMA decays to track and save, e.g. 0.9,0.99.")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="Stop after this many optimizer steps. Useful for quick regression checks.")
    parser.add_argument("--learned-path-ckpt", type=str, default=None,
                        help="Optional frozen path checkpoint for learned-path flow matching targets.")
    parser.add_argument("--learned-path-fd-step", type=float, default=0.02)
    parser.add_argument("--learned-path-mix", type=float, default=1.0,
                        help="Mix weight on the learned path. 0 keeps the linear path, 1 uses the learned path.")
    parser.add_argument("--disable-path-residual-x0-time-rho", action="store_true",
                        help="Use rho=1 for residual x0_hat sampling instead of rho=clamp(1-t, 0, 1).")
    parser.add_argument("--path-rho-constant", type=float, default=None,
                        help="Use a constant rho in [0,1] instead of the time-dependent schedule.")
    parser.add_argument("--x0-hat-rho-scale", type=float, default=1.0,
                        help="Scale in rho(t)=clamp(scale*(1-t),0,1) when constant rho is unset.")
    parser.add_argument(
        "--learned-path-subtractor-residual-scale",
        type=float,
        default=1.0,
        help="Scale factor on the dual-stem subtractor residual used when reconstructing learned-path targets.",
    )

    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for training forwards/losses. fp16 also enables GradScaler.",
    )

    parser.add_argument("--attn-func", type=str, default=None,
                        choices=["base", "fa2", "fa3", "torch_sdpa"],
                        help="Attention backend. Default: timm built-in.")

    parser.add_argument(
        "--class-dropout-prob",
        type=float,
        default=0.1,
        help="Random CFG label-dropout probability. MG reference runs use 0.0.",
    )
    parser.add_argument(
        "--learn-sigma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the existing learned-sigma head by default; MG reference runs use --no-learn-sigma.",
    )
    parser.add_argument(
        "--mg-start-step",
        type=int,
        default=-1,
        help="Step at which MG target augmentation activates. -1 preserves legacy training exactly.",
    )
    parser.add_argument(
        "--mg-data-ratio",
        type=float,
        nargs=2,
        default=[0.2, 0.1],
        metavar=("MG_FRAC", "DROP_FRAC"),
        help="MG compatibility tuple. MG_FRAC is reserved; DROP_FRAC is the unconditional fraction.",
    )
    parser.add_argument(
        "--mgw",
        type=float,
        nargs=2,
        default=[1.45, 1.45],
        metavar=("LOW", "HIGH"),
        help="Uniform range for the MG guidance weight.",
    )
    parser.add_argument(
        "--mg-data-side-threshold",
        type=float,
        default=0.75,
        help="Apply MG where t > 1 - threshold under this repository's t=1=data convention.",
    )
    parser.add_argument(
        "--mg-contrastive",
        action="store_true",
        help="Use a random non-target class instead of the unconditional slot for the reference prediction.",
    )

    parse_transport_args(parser)
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1.")
    if args.autosave_every < 0:
        raise ValueError("--autosave-every must be non-negative.")
    if args.per_gpu_batch_size is not None and args.per_gpu_batch_size < 1:
        raise ValueError("--per-gpu-batch-size must be at least 1.")
    if args.per_gpu_batch_size is not None and args.grad_accum_steps != 1:
        raise ValueError("--grad-accum-steps is auto-computed when --per-gpu-batch-size is set.")
    if args.learned_path_mix < 0.0 or args.learned_path_mix > 1.0:
        raise ValueError("--learned-path-mix must be in [0, 1].")
    if args.learned_path_fd_step <= 0.0 or args.learned_path_fd_step >= 0.5:
        raise ValueError("--learned-path-fd-step must lie in (0, 0.5).")
    if args.learned_path_subtractor_residual_scale <= 0.0:
        raise ValueError("--learned-path-subtractor-residual-scale must be positive.")
    if not np.isfinite(args.ema_decay) or args.ema_decay < 0.0 or args.ema_decay > 1.0:
        raise ValueError("--ema-decay must be in [0, 1].")
    if args.path_rho_constant is not None:
        if not 0.0 <= args.path_rho_constant <= 1.0:
            raise ValueError(f"--path-rho-constant must be in [0, 1], got {args.path_rho_constant}.")
        if args.disable_path_residual_x0_time_rho:
            raise ValueError(
                "--path-rho-constant and --disable-path-residual-x0-time-rho are mutually exclusive."
            )
    if not 0.0 <= args.x0_hat_rho_scale <= 2.0:
        raise ValueError(f"--x0-hat-rho-scale must be in [0, 2], got {args.x0_hat_rho_scale}.")
    if not 0.0 <= args.class_dropout_prob <= 1.0:
        raise ValueError("--class-dropout-prob must be in [0, 1].")
    if args.mg_start_step < -1:
        raise ValueError("--mg-start-step must be -1 or non-negative.")
    if args.mg_start_step >= 0:
        validate_model_guidance_config(
            weight_low=args.mgw[0],
            weight_high=args.mgw[1],
            drop_fraction=args.mg_data_ratio[1],
            data_side_threshold=args.mg_data_side_threshold,
        )
    main(args)
