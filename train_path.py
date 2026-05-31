# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a separate SiT-shaped path model to learn lower-energy latent-space paths
under a frozen pretrained SiT teacher field.
"""
import argparse
import csv
import logging
import os
from copy import deepcopy
from contextlib import nullcontext
from glob import glob
from time import time

import numpy as np
from PIL import Image

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models import AutoencoderKL

from sit_utils.train_utils import get_optimizer_lr, make_lr_schedule, maybe_autocast, set_optimizer_lr
from sit_utils.latent_dataset import LATENT_SCALE, PackedVAELabelDataset, PairedPackedLatentSampler, is_packed_vae_latent_dir, paired_packed_latent_original_count, resolve_packed_latent_view_mode, sample_packed_vae_latents
from models import (
    SiT_models,
    build_dual_stem_path_sit,
    is_dual_stem_path_state_dict,
    load_dual_stem_path_state_dict,
)
from path_energy import (
    FeaturePhi,
    VAEDecoderAdapter,
    combined_path_energy,
    disable_label_dropout,
    freeze_model_params,
    generalized_mixed_path,
    generalized_mixed_path_derivative_fd,
    linear_baseline_energy,
    linear_path,
    sample_path_residual_x0_hat,
    zero_sit_output_layer,
)
from sit_utils import wandb_utils
from train import update_ema


EVAL_TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
SPAN_DIAG_TIMES = (0.1, 0.9)
AUTORESUME_FILENAME = "latest_autoresume.pt"
TEST_AUTOSAVE_EXIT_CODE = 85
IMAGENET_TRAIN_IMAGE_COUNT = 1_281_167
DUAL_STEM_PATH_ARCHES = {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary", "dual_stem_direct_residual"}
DUAL_STEM_SUBTRACTOR_ARCHES = {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary"}


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def create_logger(logging_dir):
    if get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
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


def unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        if "ema" in obj:
            return obj["ema"]
        if "model" in obj:
            return obj["model"]
    return obj


def infer_learn_sigma_from_state_dict(state_dict, model_name):
    patch_size = int(model_name.split("/")[-1])
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


def load_state_dict(path):
    return unwrap_checkpoint(load_torch_checkpoint(path))


def build_path_model(args, latent_size, learn_sigma, device):
    if args.path_arch == "legacy_sit":
        model = SiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=learn_sigma,
            attn_func=args.attn_func,
        )
    elif args.path_arch in DUAL_STEM_PATH_ARCHES:
        model = build_dual_stem_path_sit(
            args.model,
            input_size=latent_size,
            num_classes=args.num_classes,
            learn_sigma=learn_sigma,
            use_endpoint_conditioning=args.path_use_endpoint_conditioning,
            teacher_residual_boundary_lambda=args.path_boundary_envelope_lambda,
            attn_func=args.attn_func,
        )
    else:
        raise ValueError(f"Unsupported --path-arch: {args.path_arch}")
    return model.to(device)


def initialize_path_model_from_state(path_model, path_init_state, args):
    if args.path_arch == "legacy_sit":
        if is_dual_stem_path_state_dict(path_init_state):
            raise ValueError(
                "Cannot initialize a legacy SiT path model from a dual-stem path checkpoint. "
                "Use --path-arch dual_stem_teacher_residual / dual_stem_teacher_residual_subboundary / dual_stem_direct_residual or provide a vanilla SiT checkpoint."
            )
        path_model.load_state_dict(path_init_state, strict=True)
        if not args.keep_path_init_output_layer:
            zero_sit_output_layer(path_model)
        return "legacy_zero_output" if not args.keep_path_init_output_layer else "legacy_preserve_output"

    if args.path_arch in {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary"}:
        if is_dual_stem_path_state_dict(path_init_state):
            path_model.load_state_dict(path_init_state, strict=True)
            return "dual_stem_checkpoint"
        load_dual_stem_path_state_dict(
            path_model,
            path_init_state,
            strict=True,
            zero_init_new_modules=True,
        )
        return "dual_stem_from_vanilla_teacher"

    if args.path_arch == "dual_stem_direct_residual":
        if is_dual_stem_path_state_dict(path_init_state):
            path_model.load_state_dict(path_init_state, strict=True)
            zero_sit_output_layer(path_model)
            return "dual_stem_direct_checkpoint_zero_output"
        load_dual_stem_path_state_dict(
            path_model,
            path_init_state,
            strict=True,
            zero_init_new_modules=True,
        )
        zero_sit_output_layer(path_model)
        return "dual_stem_direct_from_vanilla_zero_output"

    raise ValueError(f"Unsupported --path-arch: {args.path_arch}")


def should_run_zero_step_sanity(args, init_mode, path_init_matches_subtractor=True):
    if init_mode == "resume":
        return False, "resume checkpoint provided"
    if args.path_arch == "legacy_sit":
        if args.keep_path_init_output_layer:
            return False, "path-init output layer preserved"
        return True, "legacy SiT path-net with zeroed output layer"
    if args.path_arch in {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary"}:
        if init_mode == "dual_stem_from_vanilla_teacher" and path_init_matches_subtractor:
            return True, "dual-stem path-net copied from the same vanilla SiT checkpoint used by the fixed path subtractor"
        if init_mode == "dual_stem_from_vanilla_teacher":
            return False, "path-init checkpoint differs from the fixed path subtractor, so teacher-residualization is not initially linear"
        return False, "dual-stem checkpoint initialization does not imply exact linear-path startup"
    if args.path_arch == "dual_stem_direct_residual":
        return True, "dual-stem direct-residual path-net with zeroed output head"
    return False, "unsupported path arch"


def randn_like(x, seed=None):
    if seed is None:
        return torch.randn_like(x)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def repeat_batch_for_t(x0, x1, y, t):
    repeats = t.numel()
    batch_size = x0.shape[0]
    x0_rep = x0.repeat_interleave(repeats, dim=0)
    x1_rep = x1.repeat_interleave(repeats, dim=0)
    y_rep = y.repeat_interleave(repeats, dim=0)
    t_rep = t.to(device=x0.device, dtype=x0.dtype).repeat(batch_size)
    return x0_rep, x1_rep, y_rep, t_rep


def sample_random_times(batch_size, num_time_samples, device, dtype):
    # Cover the full time interval; boundary points are handled with one-sided
    # finite differences inside the path-derivative helpers.
    return torch.rand(batch_size * num_time_samples, device=device, dtype=dtype)


def build_feature_loss_modules(args, vae, device):
    if args.path_loss_mode != "feature_energy":
        return None, None
    return VAEDecoderAdapter(vae), FeaturePhi("InceptionV3", device)


def diagnostic_speed_time_points(fd_step):
    candidates = (
        ("edge_lo", fd_step),
        ("inner_lo", 0.1),
        ("mid", 0.5),
        ("inner_hi", 0.9),
        ("edge_hi", 1.0 - fd_step),
    )
    seen = set()
    points = []
    for label, value in candidates:
        rounded_value = round(float(value), 6)
        if rounded_value in seen:
            continue
        seen.add(rounded_value)
        points.append((label, float(value)))
    return points


def metric_means(path_metrics, linear_metrics):
    path_total = path_metrics["total"].mean()
    path_track = path_metrics["track"].mean()
    path_euclid = path_metrics["euclid"].mean()
    path_feature_energy = path_metrics["feature_energy"].mean()
    path_accel = path_metrics["accel"].mean()
    linear_total = linear_metrics["total"].mean()
    linear_track = linear_metrics["track"].mean()
    linear_euclid = linear_metrics["euclid"].mean()
    linear_feature_energy = linear_metrics["feature_energy"].mean()
    linear_accel = linear_metrics["accel"].mean()
    delta_total = linear_total - path_total
    delta_track = linear_track - path_track
    relative_total_gain = delta_total / torch.clamp(linear_total, min=1e-8)
    return {
        "path_total": path_total,
        "path_track": path_track,
        "path_euclid": path_euclid,
        "path_feature_energy": path_feature_energy,
        "path_accel": path_accel,
        "linear_total": linear_total,
        "linear_track": linear_track,
        "linear_euclid": linear_euclid,
        "linear_feature_energy": linear_feature_energy,
        "linear_accel": linear_accel,
        "delta_total": delta_total,
        "delta_track": delta_track,
        "relative_total_gain": relative_total_gain,
    }


def init_metric_sums():
    return {
        "path_total": 0.0,
        "path_track": 0.0,
        "path_euclid": 0.0,
        "path_feature_energy": 0.0,
        "path_accel": 0.0,
        "linear_total": 0.0,
        "linear_track": 0.0,
        "linear_euclid": 0.0,
        "linear_feature_energy": 0.0,
        "linear_accel": 0.0,
        "delta_total": 0.0,
        "delta_track": 0.0,
        "relative_total_gain": 0.0,
    }


def update_metric_sums(metric_sums, metric_tensors, count):
    for key in metric_sums:
        metric_sums[key] += metric_tensors[key].detach().item() * count


def reduce_metric_sums(metric_sums, count, device):
    keys = list(metric_sums.keys())
    values = torch.tensor(
        [metric_sums[key] for key in keys] + [float(count)],
        device=device,
        dtype=torch.float64,
    )
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_count = max(values[-1].item(), 1.0)
    return {key: (values[idx].item() / total_count) for idx, key in enumerate(keys)}


def compute_geometry_diagnostics(
    path_model,
    teacher,
    x0,
    x1,
    y,
    h,
    learned_mix=1.0,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    disable_path_residual_x0_time_rho=False,
    x0_hat_rho_scale=1.0,
):
    batch_size = x0.shape[0]

    span_t = torch.tensor(SPAN_DIAG_TIMES, device=x0.device, dtype=x0.dtype)
    x0_rep, x1_rep, y_rep, t_rep = repeat_batch_for_t(x0, x1, y, span_t)
    x0_hat_rep = sample_path_residual_x0_hat(
        x0_rep,
        t_rep,
        disable_path_residual_x0_time_rho=disable_path_residual_x0_time_rho,
        x0_hat_rho_scale=x0_hat_rho_scale,
    )
    path_gamma = generalized_mixed_path(
        path_model,
        teacher,
        x0_rep,
        x1_rep,
        y_rep,
        t_rep,
        learned_mix=learned_mix,
        path_parameterization=path_parameterization,
        path_subtractor=path_subtractor,
        subtractor_residual_scale=subtractor_residual_scale,
        x0_hat=x0_hat_rep,
    )
    linear_gamma = linear_path(x0_rep, x1_rep, t_rep)
    path_gamma = path_gamma.reshape(batch_size, span_t.numel(), -1)
    linear_gamma = linear_gamma.reshape(batch_size, span_t.numel(), -1)

    path_span = (path_gamma[:, 1] - path_gamma[:, 0]).norm(dim=1).mean()
    linear_span = (linear_gamma[:, 1] - linear_gamma[:, 0]).norm(dim=1).mean()

    speed_points = diagnostic_speed_time_points(h)
    speed_t = torch.tensor([value for _, value in speed_points], device=x0.device, dtype=x0.dtype)
    x0_rep, x1_rep, y_rep, t_rep = repeat_batch_for_t(x0, x1, y, speed_t)
    x0_hat_rep = sample_path_residual_x0_hat(
        x0_rep,
        t_rep,
        disable_path_residual_x0_time_rho=disable_path_residual_x0_time_rho,
        x0_hat_rho_scale=x0_hat_rho_scale,
    )
    _, path_gamma_dot = generalized_mixed_path_derivative_fd(
        path_model,
        teacher,
        x0_rep,
        x1_rep,
        y_rep,
        t_rep,
        h,
        learned_mix=learned_mix,
        path_parameterization=path_parameterization,
        path_subtractor=path_subtractor,
        subtractor_residual_scale=subtractor_residual_scale,
        x0_hat=x0_hat_rep,
    )
    linear_gamma_dot = x1_rep - x0_rep
    path_speed_rms = path_gamma_dot.reshape(batch_size, speed_t.numel(), -1).pow(2).mean(dim=2).sqrt()
    linear_speed_rms = linear_gamma_dot.reshape(batch_size, speed_t.numel(), -1).pow(2).mean(dim=2).sqrt()

    diagnostics = {
        "path_span_0p1_0p9": path_span,
        "linear_span_0p1_0p9": linear_span,
    }
    for idx, (label, _) in enumerate(speed_points):
        path_speed = path_speed_rms[:, idx].mean()
        linear_speed = linear_speed_rms[:, idx].mean()
        diagnostics[f"path_speed_rms_{label}"] = path_speed
        diagnostics[f"linear_speed_rms_{label}"] = linear_speed
    return diagnostics


def finalize_geometry_diagnostics(diagnostics, fd_step):
    finalized = dict(diagnostics)
    finalized["span_ratio_0p1_0p9"] = finalized["path_span_0p1_0p9"] / max(finalized["linear_span_0p1_0p9"], 1e-8)
    for label, _ in diagnostic_speed_time_points(fd_step):
        finalized[f"speed_rms_ratio_{label}"] = (
            finalized[f"path_speed_rms_{label}"] / max(finalized[f"linear_speed_rms_{label}"], 1e-8)
        )
    return finalized


def format_geometry_diagnostics(diagnostics, fd_step):
    speed_parts = []
    for label, value in diagnostic_speed_time_points(fd_step):
        speed_parts.append(
            f"t={value:.2f}: "
            f"path={diagnostics[f'path_speed_rms_{label}']:.6f}, "
            f"linear={diagnostics[f'linear_speed_rms_{label}']:.6f}, "
            f"ratio={diagnostics[f'speed_rms_ratio_{label}']:.6f}"
        )
    speed_log = " | ".join(speed_parts)
    return (
        f"span_0.1_0.9 path={diagnostics['path_span_0p1_0p9']:.6f}, "
        f"linear={diagnostics['linear_span_0p1_0p9']:.6f}, "
        f"ratio={diagnostics['span_ratio_0p1_0p9']:.6f}; "
        f"speed_rms {speed_log}"
    )


def build_eval_split(dataset_len, global_batch_size, eval_num_batches, seed):
    eval_size = global_batch_size * eval_num_batches
    if dataset_len <= eval_size:
        raise ValueError(
            f"Dataset of size {dataset_len} is too small for eval split of size {eval_size}."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    return train_indices, eval_indices


def encode_to_latents(vae, x, *, inputs_are_packed_latents=False):
    with torch.no_grad():
        if inputs_are_packed_latents:
            return sample_packed_vae_latents(x)
        return vae.encode(x).latent_dist.sample().mul_(LATENT_SCALE)


def run_zero_step_sanity_check(path_model, teacher, path_subtractor, vae, feature_decoder, feature_phi, loader, device, args, logger, *, inputs_are_packed_latents=False):
    batch = next(iter(loader))
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad(), maybe_autocast("cuda", args.amp_dtype):
        x1 = encode_to_latents(vae, x, inputs_are_packed_latents=inputs_are_packed_latents)
        x0 = randn_like(x1, seed=args.global_seed)
        t = torch.tensor(EVAL_TIMES, device=device, dtype=x1.dtype)
        x0_rep, x1_rep, y_rep, t_rep = repeat_batch_for_t(x0, x1, y, t)
        x0_hat_rep = sample_path_residual_x0_hat(
            x0_rep,
            t_rep,
            disable_path_residual_x0_time_rho=args.disable_path_residual_x0_time_rho,
            x0_hat_rho_scale=args.x0_hat_rho_scale,
        )
        path_metrics = combined_path_energy(
            path_model,
            teacher,
            x0_rep,
            x1_rep,
            y_rep,
            t_rep,
            beta=args.beta,
            h=args.fd_step,
            learned_mix=args.learned_path_mix,
            accel_reg_weight=args.accel_reg_weight,
            path_parameterization=args.path_arch,
            loss_mode=args.path_loss_mode,
            path_subtractor=path_subtractor,
            feature_decoder=feature_decoder,
            feature_phi=feature_phi,
            feature_t_thresh=args.feature_t_thresh,
            feature_energy_scale=args.feature_energy_scale,
            feature_global_scale=args.feature_global_scale,
            x0_hat=x0_hat_rep,
        )
        linear_metrics = linear_baseline_energy(
            teacher,
            x0_rep,
            x1_rep,
            y_rep,
            t_rep,
            beta=args.beta,
            h=args.fd_step,
            loss_mode=args.path_loss_mode,
            feature_decoder=feature_decoder,
            feature_phi=feature_phi,
            feature_t_thresh=args.feature_t_thresh,
            feature_energy_scale=args.feature_energy_scale,
            feature_global_scale=args.feature_global_scale,
        )

    diffs = {
        "gamma": (path_metrics["gamma_t"] - linear_metrics["gamma_t"]).abs().max(),
        "gamma_dot": (path_metrics["gamma_dot_t"] - linear_metrics["gamma_dot_t"]).abs().max(),
        "total": (path_metrics["total"] - linear_metrics["total"]).abs().max(),
        "track": (path_metrics["track"] - linear_metrics["track"]).abs().max(),
        "euclid": (path_metrics["euclid"] - linear_metrics["euclid"]).abs().max(),
        "feature_energy": (path_metrics["feature_energy"] - linear_metrics["feature_energy"]).abs().max(),
        "accel": (path_metrics["accel"] - linear_metrics["accel"]).abs().max(),
    }
    diff_tensor = torch.tensor([value.item() for value in diffs.values()], device=device)
    if is_distributed():
        dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)

    if get_rank() == 0:
        keys = list(diffs.keys())
        diff_log = ", ".join(f"{key}_max_abs={diff_tensor[idx].item():.3e}" for idx, key in enumerate(keys))
        logger.info(f"Zero-step sanity check: {diff_log}")

    if diff_tensor.max().item() > args.sanity_tol:
        if args.path_loss_mode == "feature_energy":
            return
        logger.warning(
            f"Zero-step sanity check failed: max deviation {diff_tensor.max().item():.3e} "
            f"exceeds tolerance {args.sanity_tol:.3e}"
        )
        # raise RuntimeError(
        #     f"Zero-step sanity check failed: max deviation {diff_tensor.max().item():.3e} "
        #     f"exceeds tolerance {args.sanity_tol:.3e}"
        # )


def evaluate(path_model, teacher, path_subtractor, vae, feature_decoder, feature_phi, loader, device, args, *, inputs_are_packed_latents=False,):
    was_training = path_model.training
    path_model.eval()
    teacher.eval()
    if path_subtractor is not None:
        path_subtractor.eval()

    metric_sums = init_metric_sums()
    total_count = 0
    diagnostic_sums = None
    diagnostic_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= args.eval_num_batches:
                break
            x = x.to(device)
            y = y.to(device)
            with maybe_autocast("cuda", args.amp_dtype):
                x1 = encode_to_latents(vae, x, inputs_are_packed_latents=inputs_are_packed_latents)
                seed = args.global_seed + 100_000 + batch_idx * get_world_size() + get_rank()
                x0 = randn_like(x1, seed=seed)
                t = torch.tensor(EVAL_TIMES, device=device, dtype=x1.dtype)
                x0_rep, x1_rep, y_rep, t_rep = repeat_batch_for_t(x0, x1, y, t)
                x0_hat_rep = sample_path_residual_x0_hat(
                    x0_rep,
                    t_rep,
                    disable_path_residual_x0_time_rho=args.disable_path_residual_x0_time_rho,
                    x0_hat_rho_scale=args.x0_hat_rho_scale,
                )

                path_metrics = combined_path_energy(
                    path_model,
                    teacher,
                    x0_rep,
                    x1_rep,
                    y_rep,
                    t_rep,
                    beta=args.beta,
                    h=args.fd_step,
                    learned_mix=args.learned_path_mix,
                    accel_reg_weight=args.accel_reg_weight,
                    path_parameterization=args.path_arch,
                    loss_mode=args.path_loss_mode,
                    path_subtractor=path_subtractor,
                    feature_decoder=feature_decoder,
                    feature_phi=feature_phi,
                    feature_t_thresh=args.feature_t_thresh,
                    feature_energy_scale=args.feature_energy_scale,
                    feature_global_scale=args.feature_global_scale,
                    x0_hat=x0_hat_rep,
                )
                linear_metrics = linear_baseline_energy(
                    teacher,
                    x0_rep,
                    x1_rep,
                    y_rep,
                    t_rep,
                    beta=args.beta,
                    h=args.fd_step,
                    loss_mode=args.path_loss_mode,
                    feature_decoder=feature_decoder,
                    feature_phi=feature_phi,
                    feature_t_thresh=args.feature_t_thresh,
                    feature_energy_scale=args.feature_energy_scale,
                    feature_global_scale=args.feature_global_scale,
                )
                batch_metrics = metric_means(path_metrics, linear_metrics)
            count = x0_rep.shape[0]
            update_metric_sums(metric_sums, batch_metrics, count)
            total_count += count

            if batch_idx == 0:
                with maybe_autocast("cuda", args.amp_dtype):
                    batch_diagnostics = compute_geometry_diagnostics(
                        path_model,
                        teacher,
                        x0,
                        x1,
                        y,
                        args.fd_step,
                        learned_mix=args.learned_path_mix,
                        path_parameterization=args.path_arch,
                        path_subtractor=path_subtractor,
                        disable_path_residual_x0_time_rho=args.disable_path_residual_x0_time_rho,
                        x0_hat_rho_scale=args.x0_hat_rho_scale,
                    )
                diagnostic_sums = {key: 0.0 for key in batch_diagnostics}
                update_metric_sums(diagnostic_sums, batch_diagnostics, x0.shape[0])
                diagnostic_count += x0.shape[0]

    reduced = reduce_metric_sums(metric_sums, total_count, device)
    reduced_diagnostics = {}
    if diagnostic_sums is not None:
        reduced_diagnostics = reduce_metric_sums(diagnostic_sums, diagnostic_count, device)
        reduced_diagnostics = finalize_geometry_diagnostics(reduced_diagnostics, args.fd_step)

    if was_training:
        path_model.train()
    return reduced, reduced_diagnostics


def save_checkpoint(checkpoint_dir, train_steps, epoch, steps_in_epoch, path_model, path_ema, opt, scaler, args):
    checkpoint = {
        "model": unwrap_model(path_model).state_dict(),
        "ema": path_ema.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "steps_in_epoch": steps_in_epoch,
        "train_steps": train_steps,
    }
    if scaler.is_enabled():
        checkpoint["scaler"] = scaler.state_dict()
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    model_only_path = f"{checkpoint_dir}/{train_steps:07d}_path.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(unwrap_model(path_model).state_dict(), model_only_path)
    return checkpoint_path


def init_metrics_tsv(metrics_path):
    fieldnames = [
        "step",
        "split",
        "path_total",
        "path_track",
        "path_euclid",
        "path_feature_energy",
        "path_accel",
        "linear_total",
        "linear_track",
        "linear_euclid",
        "linear_feature_energy",
        "linear_accel",
        "delta_total",
        "delta_track",
        "relative_total_gain",
        "steps_per_sec",
    ]
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def append_metrics_tsv(metrics_path, step, split, metrics, steps_per_sec=None):
    row = {
        "step": step,
        "split": split,
        "path_total": metrics["path_total"],
        "path_track": metrics["path_track"],
        "path_euclid": metrics["path_euclid"],
        "path_feature_energy": metrics["path_feature_energy"],
        "path_accel": metrics["path_accel"],
        "linear_total": metrics["linear_total"],
        "linear_track": metrics["linear_track"],
        "linear_euclid": metrics["linear_euclid"],
        "linear_feature_energy": metrics["linear_feature_energy"],
        "linear_accel": metrics["linear_accel"],
        "delta_total": metrics["delta_total"],
        "delta_track": metrics["delta_track"],
        "relative_total_gain": metrics["relative_total_gain"],
        "steps_per_sec": "" if steps_per_sec is None else steps_per_sec,
    }
    with open(metrics_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter="\t")
        writer.writerow(row)


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")

    world_size = get_world_size()
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    if args.per_gpu_batch_size is not None:
        local_batch_size = args.per_gpu_batch_size
        global_micro_batch_size = local_batch_size * world_size
        if args.global_batch_size % global_micro_batch_size != 0:
            raise ValueError(
                "--global-batch-size must be divisible by WORLD_SIZE * --per-gpu-batch-size."
            )
        grad_accum_steps = args.global_batch_size // global_micro_batch_size
        global_batch_size = args.global_batch_size
    else:
        assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
        local_batch_size = args.global_batch_size // world_size
        global_micro_batch_size = local_batch_size * world_size
        grad_accum_steps = 1
        global_batch_size = global_micro_batch_size
    args.per_gpu_batch_size = local_batch_size
    args.grad_accum_steps = grad_accum_steps
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")
    print(
        f"Starting rank={rank}, seed={seed}, world_size={world_size}, "
        f"per_gpu_batch_size={local_batch_size}, grad_accum_steps={grad_accum_steps}."
    )

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        accel_suffix = f"-accel-{args.accel_reg_weight:.3f}" if args.accel_reg_weight > 0.0 else ""
        loss_mode_suffix = "" if args.path_loss_mode == "track_mixed_euclid_learned" else f"-loss-{args.path_loss_mode.replace('_', '-')}"
        feature_loss_suffix = ""
        if args.path_loss_mode == "feature_energy":
            feature_loss_suffix = f"-fthr-{args.feature_t_thresh:.3f}-fscale-{args.feature_energy_scale:.3f}-fglob-{args.feature_global_scale:.3f}"
        path_arch_tag_map = {
            "legacy_sit": "legacy",
            "dual_stem_teacher_residual": "dualstem-resid",
            "dual_stem_teacher_residual_subboundary": "dualstem-subboundary",
            "dual_stem_direct_residual": "dualstem-directresid",
        }
        path_arch_tag = path_arch_tag_map[args.path_arch]
        endpoint_tag = ""
        if args.path_arch in DUAL_STEM_PATH_ARCHES:
            endpoint_tag = "-endcond" if args.path_use_endpoint_conditioning else "-noendcond"
        proposed_experiment_name = (
            f"{experiment_index:03d}-path-{model_string_name}-{path_arch_tag}{endpoint_tag}-"
            f"beta-{args.beta:.3f}-fd-{args.fd_step:.3f}{accel_suffix}{loss_mode_suffix}{feature_loss_suffix}"
        )
        experiment_dir, is_new_experiment = resolve_experiment_dir(
            args.results_dir,
            proposed_experiment_name,
        )
        experiment_name = os.path.basename(experiment_dir)
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        metrics_path = f"{experiment_dir}/metrics.tsv"
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if is_new_experiment or not os.path.exists(metrics_path):
            init_metrics_tsv(metrics_path)
        logger = create_logger(experiment_dir)
        if is_new_experiment:
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            logger.info(f"Reusing experiment directory at {experiment_dir}")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    if is_distributed():
        dist.barrier()
    if rank != 0:
        experiment_dir = latest_experiment_dir(args.results_dir)
        if experiment_dir is None:
            raise RuntimeError(f"No experiment directory found under {args.results_dir}")
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        metrics_path = f"{experiment_dir}/metrics.tsv"
        logger = create_logger(None)

    logger.info(f"Path model family: {args.model}")
    logger.info(f"Teacher model family: {args.teacher_model}")
    logger.info(f"Path-net EMA decay: {args.path_net_ema:.6f}")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    uses_path_subtractor = args.path_arch in DUAL_STEM_SUBTRACTOR_ARCHES

    energy_teacher_state = load_state_dict(args.teacher_ckpt)
    energy_teacher_learn_sigma = infer_learn_sigma_from_state_dict(energy_teacher_state, args.teacher_model)
    path_subtractor_ckpt = args.path_subtractor_ckpt or args.teacher_ckpt
    if uses_path_subtractor:
        args.path_subtractor_ckpt = path_subtractor_ckpt
    path_subtractor_state = None
    path_subtractor_learn_sigma = None
    if uses_path_subtractor:
        path_subtractor_state = load_state_dict(path_subtractor_ckpt)
        path_subtractor_learn_sigma = infer_learn_sigma_from_state_dict(path_subtractor_state, args.model)

    path_init_ckpt = args.path_init_ckpt or (path_subtractor_ckpt if uses_path_subtractor else args.teacher_ckpt)
    path_init_matches_subtractor = uses_path_subtractor and (
        os.path.abspath(path_init_ckpt) == os.path.abspath(path_subtractor_ckpt)
    )
    path_init_state = load_state_dict(path_init_ckpt)
    path_learn_sigma = infer_learn_sigma_from_state_dict(path_init_state, args.model)
    if energy_teacher_learn_sigma != path_learn_sigma:
        raise ValueError("Energy-teacher checkpoint and path init checkpoint disagree on learn_sigma.")
    if uses_path_subtractor and path_subtractor_learn_sigma != path_learn_sigma:
        raise ValueError("Path-subtractor checkpoint and path init checkpoint disagree on learn_sigma.")

    teacher = SiT_models[args.teacher_model](input_size=latent_size, num_classes=args.num_classes, learn_sigma=energy_teacher_learn_sigma, attn_func=args.attn_func).to(device)
    teacher.load_state_dict(energy_teacher_state, strict=True)
    freeze_model_params(teacher)
    teacher.eval()

    if not uses_path_subtractor:
        path_subtractor = None
    elif os.path.abspath(path_subtractor_ckpt) == os.path.abspath(args.teacher_ckpt) and args.model == args.teacher_model:
        path_subtractor = teacher
    else:
        path_subtractor = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes, learn_sigma=path_subtractor_learn_sigma, attn_func=args.attn_func).to(device)
        path_subtractor.load_state_dict(path_subtractor_state, strict=True)
        freeze_model_params(path_subtractor)
        path_subtractor.eval()

    path_model = build_path_model(args, latent_size, path_learn_sigma, device)
    path_ema = deepcopy(path_model).to(device)
    freeze_model_params(path_ema)
    opt = torch.optim.AdamW(
        path_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    start_steps_in_epoch = 0
    train_steps = 0
    init_mode = None
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
    if resume_path is not None:
        logger.info(f"Loading training state from {resume_path} ({resume_source})")
        resume_obj = load_torch_checkpoint(resume_path)
        resume_model_state = resume_obj["model"] if isinstance(resume_obj, dict) and "model" in resume_obj else unwrap_checkpoint(resume_obj)
        path_model.load_state_dict(resume_model_state, strict=True)
        if isinstance(resume_obj, dict) and "ema" in resume_obj:
            path_ema.load_state_dict(resume_obj["ema"], strict=True)
        else:
            update_ema(path_ema, path_model, decay=0)
        if "opt" in resume_obj and not args.reset_opt_on_resume:
            opt.load_state_dict(resume_obj["opt"])
            if "scaler" in resume_obj and scaler.is_enabled():
                scaler.load_state_dict(resume_obj["scaler"])
        start_epoch = resume_obj.get("epoch", 0)
        start_steps_in_epoch = resume_obj.get("steps_in_epoch", 0)
        train_steps = resume_obj.get("train_steps", 0)
        init_mode = "resume"
    else:
        init_mode = initialize_path_model_from_state(path_model, path_init_state, args)
        update_ema(path_ema, path_model, decay=0)
    set_optimizer_lr(opt, args.lr)

    disable_label_dropout(path_model)
    path_ema.eval()
    if is_distributed():
        logger.info(f"DISTRIBUTED is enabled, wrapping path_model in DDP")
        path_model = DDP(path_model, device_ids=[device])

    use_packed_latents = is_packed_vae_latent_dir(args.data_path)
    packed_latent_view_mode = "all"
    vae = None
    if (not use_packed_latents) or args.path_loss_mode == "feature_energy":
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        vae.enable_gradient_checkpointing()
    feature_decoder, feature_phi = build_feature_loss_modules(args, vae, device)
    logger.info(f"Energy teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    if path_subtractor is not None:
        logger.info(f"Path subtractor parameters: {sum(p.numel() for p in path_subtractor.parameters()):,}")
        logger.info(f"Path Parameters: {sum(p.numel() for p in path_model.parameters()):,}")
        logger.info(f"Path architecture: {args.path_arch}")
        if uses_path_subtractor:
            endpoint_status = "enabled" if args.path_use_endpoint_conditioning else "disabled"
            logger.info(f"Dual-stem endpoint conditioning: {endpoint_status}; initialization mode: {init_mode}")
            if args.path_arch == "dual_stem_teacher_residual_subboundary":
                logger.info("Boundary enforcement: subtraction-based endpoint correction (gamma = gamma_lin + c(t) - (1-t)c(0) - t c(1)).")
            else:
                logger.info(
                    "Boundary enforcement: multiplicative teacher-residual envelope "
                    f"w_lambda(t)=(t(1-t))^lambda * 0.25^(1-lambda), with lambda={args.path_boundary_envelope_lambda:.6f}."
                )
            logger.info(f"Energy teacher checkpoint: {args.teacher_ckpt}")
            logger.info(f"Path subtractor checkpoint: {path_subtractor_ckpt}")
            logger.info(f"Path-init checkpoint matches path subtractor: {path_init_matches_subtractor}")
            if (init_mode == "dual_stem_from_vanilla_teacher") and (not path_init_matches_subtractor):
                logger.info(
                    "Because --path-init-ckpt differs from --path-subtractor-ckpt, the dual-stem teacher-residual path "
                    "does not start exactly at the linear interpolation even though the new modules are zero-initialized."
                )
        if args.keep_path_init_output_layer:
            logger.info(
                "Ignoring --keep-path-init-output-layer for dual-stem teacher-residual paths: "
                "exact linear initialization comes from teacher-residualization plus zeroed new modules, not from zeroing the final head."
            )
    elif args.path_arch == "dual_stem_direct_residual":
        logger.info(f"Path Parameters: {sum(p.numel() for p in path_model.parameters()):,}")
        logger.info(f"Path architecture: {args.path_arch}")
        endpoint_status = "enabled" if args.path_use_endpoint_conditioning else "disabled"
        logger.info(f"Dual-stem endpoint conditioning: {endpoint_status}; initialization mode: {init_mode}")
        logger.info(
            "Boundary enforcement: multiplicative direct-residual envelope "
            f"gamma = gamma_lin + w_lambda(t) * f_phi(gamma_lin_hat, delta_hat, t, y), with lambda={args.path_boundary_envelope_lambda:.6f}."
        )
        logger.info(f"Energy teacher checkpoint: {args.teacher_ckpt}")
        logger.info("Path subtractor: ignored for dual_stem_direct_residual.")
        logger.info("subtractor_residual_scale is ignored for dual_stem_direct_residual because no subtractor is used.")
        if args.path_subtractor_ckpt is not None:
            logger.info(f"Ignoring --path-subtractor-ckpt for dual_stem_direct_residual: {args.path_subtractor_ckpt}")
        if args.keep_path_init_output_layer:
            logger.info(
                "Ignoring --keep-path-init-output-layer for dual_stem_direct_residual: "
                "fresh path-stage initialization zeroes the final output head."
            )
    else:
        logger.info(f"Legacy path initialization mode: {init_mode}")
    if args.learned_path_mix != 1.0:
        logger.info(f"Using mixed-path training/eval with learned_path_mix={args.learned_path_mix:.3f}")
    logger.info(f"Using path loss mode: {args.path_loss_mode}")
    if args.path_loss_mode == "feature_energy":
        logger.info(
            "Feature-energy loss uses mixed-path latent euclid plus InceptionV3 feature energy "
            f"with t_thresh={args.feature_t_thresh:.6f}, scale={args.feature_energy_scale:.6f}, "
            f"and global_scale={args.feature_global_scale:.6f}."
        )
        logger.info("In feature_energy mode, the logged euclid metric is the latent-space velocity energy and the logged feature_energy metric is the masked feature-space energy.")
        logger.info("Ignoring --beta for feature_energy because that loss mode uses only the euclid/feature blend.")
        if args.accel_reg_weight > 0.0:
            logger.info("Ignoring --accel-reg-weight for feature_energy because that loss mode uses only the euclid/feature blend.")
    elif args.accel_reg_weight > 0.0:
        logger.info(f"Using second-order acceleration regularization with weight={args.accel_reg_weight:.6f}")

    if use_packed_latents:
        train_dataset_full = PackedVAELabelDataset(args.data_path)
        eval_dataset_full = PackedVAELabelDataset(args.data_path)
        packed_latent_view_mode = resolve_packed_latent_view_mode(args.packed_latent_view_mode, len(train_dataset_full), expected_original_count=IMAGENET_TRAIN_IMAGE_COUNT, tolerance=global_batch_size)
        logger.info(
            f"Using packed VAE latents from {args.data_path}; "
            "sampling final 4-channel latents from stored mean/std at runtime."
        )
    else:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        eval_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        train_dataset_full = ImageFolder(args.data_path, transform=train_transform)
        eval_dataset_full = ImageFolder(args.data_path, transform=eval_transform)
        logger.info(
            f"Using image dataset from {args.data_path}; encoding images to latents with the VAE at runtime."
        )
    train_sampler = None
    eval_sampler = None
    if use_packed_latents and packed_latent_view_mode == "one-per-image":
        num_original_images = paired_packed_latent_original_count(len(train_dataset_full))
        train_indices, eval_indices = build_eval_split(num_original_images, global_micro_batch_size, args.eval_num_batches, args.global_seed)
        train_dataset = train_dataset_full
        eval_dataset = eval_dataset_full
        train_size_for_log = len(train_indices)
        eval_size_for_log = len(eval_indices)
        train_sampler = PairedPackedLatentSampler(original_indices=train_indices, num_replicas=get_world_size(), rank=rank, shuffle=True, seed=args.global_seed, view_policy="random")
        eval_sampler = PairedPackedLatentSampler(original_indices=eval_indices, num_replicas=get_world_size(), rank=rank, shuffle=False, seed=args.global_seed, drop_last=True, view_policy="first")
    else:
        train_indices, eval_indices = build_eval_split(
            len(train_dataset_full),
            global_micro_batch_size,
            args.eval_num_batches,
            args.global_seed,
        )
        train_dataset = Subset(train_dataset_full, train_indices)
        eval_dataset = Subset(eval_dataset_full, eval_indices)
        train_size_for_log = len(train_dataset)
        eval_size_for_log = len(eval_dataset)
        if is_distributed():
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=rank,
                shuffle=True,
                seed=args.global_seed,
            )
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=get_world_size(),
                rank=rank,
                shuffle=False,
                seed=args.global_seed,
                drop_last=True,
            )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if use_packed_latents and packed_latent_view_mode == "one-per-image":
        num_original_images = paired_packed_latent_original_count(len(train_dataset_full))
        logger.info(
            f"Dataset contains {len(train_dataset_full):,} packed latent views ({args.data_path}); "
            f"packed-latent view mode resolved to one-per-image from --packed-latent-view-mode={args.packed_latent_view_mode}, "
            f"so train/eval split is applied over {num_original_images:,} original images: "
            f"train={train_size_for_log:,}, eval={eval_size_for_log:,}."
        )
    else:
        logger.info(
            f"Dataset contains {len(train_dataset_full):,} "
            f"{'packed latent samples' if use_packed_latents else 'images'} ({args.data_path}); "
            f"train={train_size_for_log:,}, eval={eval_size_for_log:,}"
        )
    logger.info(
        f"Using per_gpu_batch_size={local_batch_size}, "
        f"global_batch_size={global_batch_size}, "
        f"grad_accum_steps={grad_accum_steps}"
    )
    logger.info(f"Attention backend: {args.attn_func or 'default'}, autocast: {args.amp_dtype}")

    steps_per_epoch = len(train_loader) // grad_accum_steps
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

    if init_mode == "resume":
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

    path_model.train()
    teacher.eval()
    if path_subtractor is not None:
        path_subtractor.eval()

    run_sanity, sanity_reason = should_run_zero_step_sanity(
        args,
        init_mode,
        path_init_matches_subtractor=path_init_matches_subtractor,
    )
    if run_sanity:
        logger.info(f"Running zero-step sanity check ({sanity_reason}).")
        run_zero_step_sanity_check(
            path_model,
            teacher,
            path_subtractor,
            vae,
            feature_decoder,
            feature_phi,
            train_loader,
            device,
            args,
            logger,
            inputs_are_packed_latents=use_packed_latents,
        )
    else:
        logger.info(f"Skipping zero-step sanity check ({sanity_reason}).")

    running_metric_sums = init_metric_sums()
    running_count = 0
    log_steps = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    resume_steps_in_epoch = start_steps_in_epoch
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if eval_sampler is not None:
            eval_sampler.set_epoch(0)
        epoch_steps_completed = resume_steps_in_epoch if epoch == start_epoch else 0
        logger.info(
            f"Beginning epoch {epoch}..."
            if epoch_steps_completed == 0
            else f"Beginning epoch {epoch} at resumed optimizer-step offset {epoch_steps_completed}..."
        )
        step_metric_sums = init_metric_sums()
        step_count = 0
        start_batch_idx = epoch_steps_completed * grad_accum_steps
        train_loader_iter = iter(train_loader)
        fast_forward_loader(
            train_loader_iter,
            start_batch_idx,
            logger=logger,
            epoch=epoch,
            train_steps=train_steps,
        )
        resume_steps_in_epoch = 0
        for batch_idx, (x, y) in enumerate(train_loader_iter, start=start_batch_idx):
            if epoch_steps_completed >= steps_per_epoch:
                break
            if batch_idx % grad_accum_steps == 0:
                if lr_schedule is not None:
                    set_optimizer_lr(opt, lr_schedule(train_steps))
                opt.zero_grad()
            x = x.to(device)
            y = y.to(device)
            with maybe_autocast("cuda", args.amp_dtype):
                x1 = encode_to_latents(vae, x, inputs_are_packed_latents=use_packed_latents)
                x0 = randn_like(x1)
                t = sample_random_times(
                    batch_size=x1.shape[0],
                    num_time_samples=args.num_time_samples,
                    device=device,
                    dtype=x1.dtype,
                )
                if args.num_time_samples == 1:
                    x0_rep, x1_rep, y_rep, t_rep = x0, x1, y, t
                else:
                    x0_rep = x0.repeat_interleave(args.num_time_samples, dim=0)
                    x1_rep = x1.repeat_interleave(args.num_time_samples, dim=0)
                    y_rep = y.repeat_interleave(args.num_time_samples, dim=0)
                    t_rep = t
                x0_hat_rep = sample_path_residual_x0_hat(
                    x0_rep,
                    t_rep,
                    disable_path_residual_x0_time_rho=args.disable_path_residual_x0_time_rho,
                    x0_hat_rho_scale=args.x0_hat_rho_scale,
                )

                path_metrics = combined_path_energy(
                    path_model,
                    teacher,
                    x0_rep,
                    x1_rep,
                    y_rep,
                    t_rep,
                    beta=args.beta,
                    h=args.fd_step,
                    learned_mix=args.learned_path_mix,
                    accel_reg_weight=args.accel_reg_weight,
                    path_parameterization=args.path_arch,
                    loss_mode=args.path_loss_mode,
                    path_subtractor=path_subtractor,
                    feature_decoder=feature_decoder,
                    feature_phi=feature_phi,
                    feature_t_thresh=args.feature_t_thresh,
                    feature_energy_scale=args.feature_energy_scale,
                    feature_global_scale=args.feature_global_scale,
                    x0_hat=x0_hat_rep,
                )
                loss = path_metrics["total"].mean()

            should_step = (batch_idx + 1) % grad_accum_steps == 0
            sync_context = (
                path_model.no_sync()
                if is_distributed() and grad_accum_steps > 1 and not should_step
                else nullcontext()
            )
            with sync_context:
                scaled_loss = scaler.scale(loss / grad_accum_steps) if scaler.is_enabled() else (loss / grad_accum_steps)
                scaled_loss.backward()

            with torch.no_grad(), maybe_autocast("cuda", args.amp_dtype):
                linear_metrics = linear_baseline_energy(
                    teacher,
                    x0_rep,
                    x1_rep,
                    y_rep,
                    t_rep,
                    beta=args.beta,
                    h=args.fd_step,
                    loss_mode=args.path_loss_mode,
                    feature_decoder=feature_decoder,
                    feature_phi=feature_phi,
                    feature_t_thresh=args.feature_t_thresh,
                    feature_energy_scale=args.feature_energy_scale,
                    feature_global_scale=args.feature_global_scale,
                )
                batch_metrics = metric_means(path_metrics, linear_metrics)

            count = x0_rep.shape[0]
            update_metric_sums(step_metric_sums, batch_metrics, count)
            step_count += count

            if not should_step:
                if batch_idx == len(train_loader) - 1:
                    opt.zero_grad()
                continue

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            update_ema(path_ema, unwrap_model(path_model), decay=args.path_net_ema)
            for key in running_metric_sums:
                running_metric_sums[key] += step_metric_sums[key]
            running_count += step_count
            log_steps += 1
            train_steps += 1
            epoch_steps_completed += 1
            step_metric_sums = init_metric_sums()
            step_count = 0

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                reduced_metrics = reduce_metric_sums(running_metric_sums, running_count, device)
                peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                current_lr = get_optimizer_lr(opt)
                if rank == 0:
                    append_metrics_tsv(
                        metrics_path,
                        train_steps,
                        "train",
                        reduced_metrics,
                        steps_per_sec=steps_per_sec,
                    )
                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"path_total={reduced_metrics['path_total']:.6f}, "
                        f"path_track={reduced_metrics['path_track']:.6f}, "
                        f"path_euclid={reduced_metrics['path_euclid']:.6f}, "
                        f"path_feature_energy={reduced_metrics['path_feature_energy']:.6f}, "
                        f"path_accel={reduced_metrics['path_accel']:.6f}, "
                        f"linear_total={reduced_metrics['linear_total']:.6f}, "
                        f"linear_track={reduced_metrics['linear_track']:.6f}, "
                        f"linear_euclid={reduced_metrics['linear_euclid']:.6f}, "
                        f"linear_feature_energy={reduced_metrics['linear_feature_energy']:.6f}, "
                        f"linear_accel={reduced_metrics['linear_accel']:.6f}, "
                        f"delta_total={reduced_metrics['delta_total']:.6f}, "
                        f"delta_track={reduced_metrics['delta_track']:.6f}, "
                        f"relative_total_gain={reduced_metrics['relative_total_gain']:.6f}, "
                        f"lr={current_lr:.8f}, "
                        f"per_gpu_batch_size={local_batch_size}, "
                        f"grad_accum_steps={grad_accum_steps}, "
                        f"global_batch_size={global_batch_size}, "
                        f"peak_cuda_allocated_mib={peak_alloc_mib:.1f}, "
                        f"peak_cuda_reserved_mib={peak_reserved_mib:.1f}, "
                        f"steps_per_sec={steps_per_sec:.2f}"
                    )
                    if args.wandb:
                        wandb_utils.log(
                            {
                                "train/path_total": reduced_metrics["path_total"],
                                "train/path_track": reduced_metrics["path_track"],
                                "train/path_euclid": reduced_metrics["path_euclid"],
                                "train/path_feature_energy": reduced_metrics["path_feature_energy"],
                                "train/path_accel": reduced_metrics["path_accel"],
                                "train/linear_total": reduced_metrics["linear_total"],
                                "train/linear_track": reduced_metrics["linear_track"],
                                "train/linear_euclid": reduced_metrics["linear_euclid"],
                                "train/linear_feature_energy": reduced_metrics["linear_feature_energy"],
                                "train/linear_accel": reduced_metrics["linear_accel"],
                                "train/delta_total": reduced_metrics["delta_total"],
                                "train/delta_track": reduced_metrics["delta_track"],
                                "train/relative_total_gain": reduced_metrics["relative_total_gain"],
                                "train/lr": current_lr,
                                "train/per_gpu_batch_size": local_batch_size,
                                "train/grad_accum_steps": grad_accum_steps,
                                "train/global_batch_size": global_batch_size,
                                "train/steps_per_sec": steps_per_sec,
                            },
                            step=train_steps,
                        )
                running_metric_sums = init_metric_sums()
                running_count = 0
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
                        "model": unwrap_model(path_model).state_dict(),
                        "ema": path_ema.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "steps_in_epoch": epoch_steps_completed,
                        "train_steps": train_steps,
                    }
                    if scaler.is_enabled():
                        autoresume_checkpoint["scaler"] = scaler.state_dict()
                    autoresume_path = autoresume_checkpoint_path(checkpoint_dir)
                    atomic_torch_save(autoresume_checkpoint, autoresume_path)
                    logger.info(f"Saved autoresume checkpoint to {autoresume_path}")
                if is_distributed():
                    dist.barrier()
                maybe_exit_after_autosave(train_steps, total_train_steps, logger)

            if args.eval_every > 0 and train_steps % args.eval_every == 0:
                eval_metrics, eval_diagnostics = evaluate(
                    path_model,
                    teacher,
                    path_subtractor,
                    vae,
                    feature_decoder,
                    feature_phi,
                    eval_loader,
                    device,
                    args,
                    inputs_are_packed_latents=use_packed_latents,
                )
                if rank == 0:
                    append_metrics_tsv(metrics_path, train_steps, "eval", eval_metrics)
                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"eval_path_total={eval_metrics['path_total']:.6f}, "
                        f"eval_path_track={eval_metrics['path_track']:.6f}, "
                        f"eval_path_euclid={eval_metrics['path_euclid']:.6f}, "
                        f"eval_path_feature_energy={eval_metrics['path_feature_energy']:.6f}, "
                        f"eval_path_accel={eval_metrics['path_accel']:.6f}, "
                        f"eval_linear_total={eval_metrics['linear_total']:.6f}, "
                        f"eval_linear_track={eval_metrics['linear_track']:.6f}, "
                        f"eval_linear_euclid={eval_metrics['linear_euclid']:.6f}, "
                        f"eval_linear_feature_energy={eval_metrics['linear_feature_energy']:.6f}, "
                        f"eval_linear_accel={eval_metrics['linear_accel']:.6f}, "
                        f"eval_delta_total={eval_metrics['delta_total']:.6f}, "
                        f"eval_delta_track={eval_metrics['delta_track']:.6f}, "
                        f"eval_relative_total_gain={eval_metrics['relative_total_gain']:.6f}"
                    )
                    if eval_diagnostics:
                        logger.info(
                            f"(step={train_steps:07d}) eval_geometry_batch0 "
                            f"{format_geometry_diagnostics(eval_diagnostics, args.fd_step)}"
                        )
                    if args.wandb:
                        wandb_payload = {
                            "eval/path_total": eval_metrics["path_total"],
                            "eval/path_track": eval_metrics["path_track"],
                            "eval/path_euclid": eval_metrics["path_euclid"],
                            "eval/path_feature_energy": eval_metrics["path_feature_energy"],
                            "eval/path_accel": eval_metrics["path_accel"],
                            "eval/linear_total": eval_metrics["linear_total"],
                            "eval/linear_track": eval_metrics["linear_track"],
                            "eval/linear_euclid": eval_metrics["linear_euclid"],
                            "eval/linear_feature_energy": eval_metrics["linear_feature_energy"],
                            "eval/linear_accel": eval_metrics["linear_accel"],
                            "eval/delta_total": eval_metrics["delta_total"],
                            "eval/delta_track": eval_metrics["delta_track"],
                            "eval/relative_total_gain": eval_metrics["relative_total_gain"],
                        }
                        for key, value in eval_diagnostics.items():
                            wandb_payload[f"eval_geometry_batch0/{key}"] = value
                        wandb_utils.log(wandb_payload, step=train_steps)

            if args.ckpt_every > 0 and train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint_path = save_checkpoint(
                        checkpoint_dir,
                        train_steps,
                        epoch,
                        epoch_steps_completed,
                        path_model,
                        path_ema,
                        opt,
                        scaler,
                        args,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if is_distributed():
                    dist.barrier()

            if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                break

        if args.max_train_steps is not None and train_steps >= args.max_train_steps:
            break

    path_model.eval()
    if rank == 0:
        peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        logger.info(
            f"Peak CUDA memory: allocated_mib={peak_alloc_mib:.1f}, reserved_mib={peak_reserved_mib:.1f}"
        )
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Image dataset root or packed latent directory containing latents.bin and meta.npz.",
    )
    parser.add_argument("--results-dir", type=str, default="results_path")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2", help="Path-model/path-subtractor SiT family.")
    parser.add_argument("--teacher-model", type=str, choices=list(SiT_models.keys()), default=None, help="Frozen energy-teacher SiT family. Defaults to --model.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--per-gpu-batch-size", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--packed-latent-view-mode", type=str, default="auto", choices=["auto", "all", "one-per-image"], help="Sampling semantics for packed latent datasets. 'one-per-image' samples exactly one of the two pre-encoded views per source image each epoch and keeps eval on the canonical non-flipped view; 'all' keeps the historical 2N-record behavior; 'auto' enables one-per-image semantics for the doubled full-ImageNet packed format from preproccessing/README.md.")
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument(
        "--path-subtractor-ckpt",
        type=str,
        default=None,
        help="Optional frozen checkpoint used for the dual-stem subtractor role. Defaults to --teacher-ckpt.",
    )
    parser.add_argument("--path-init-ckpt", type=str, default=None)
    parser.add_argument(
        "--path-arch",
        type=str,
        default="legacy_sit",
        choices=["legacy_sit", "dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary", "dual_stem_direct_residual"],
        help=(
            "Path-model architecture / parameterization. 'legacy_sit' preserves the original path-net setup; "
            "'dual_stem_teacher_residual' uses a dual-stem SiT with the original t(1-t) teacher-residual path; "
            "'dual_stem_teacher_residual_subboundary' uses the same dual-stem SiT trunk but enforces endpoints via subtraction-based boundary correction; "
            "'dual_stem_direct_residual' uses the same dual-stem SiT trunk with gamma = gamma_lin + w(t) * f_phi(gamma_lin_hat, delta_hat, t, y)."
        ),
    )
    parser.add_argument(
        "--path-use-endpoint-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the optional pooled-delta endpoint conditioning branch in the dual-stem path model.",
    )
    parser.add_argument("--keep-path-init-output-layer", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from checkpoints/latest_autoresume.pt in the reused experiment directory when present.",
    )
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-anneal-steps", type=int, default=None,
                        help="Number of steps over which to anneal the LR starting from the current train step.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--path_net_ema", "--path-net-ema", dest="path_net_ema", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--fd-step", type=float, default=0.02)
    parser.add_argument("--path-boundary-envelope-lambda", type=float, default=1.0)
    parser.add_argument("--accel-reg-weight", type=float, default=0.0)
    parser.add_argument(
        "--path-loss-mode",
        type=str,
        default="track_mixed_euclid_learned", choices=["track_mixed_euclid_learned", "feature_energy"],
    )
    parser.add_argument("--feature-t-thresh", type=float, default=0.0)
    parser.add_argument("--feature-energy-scale", type=float, default=1.0)
    parser.add_argument("--feature-global-scale", type=float, default=1.0)
    parser.add_argument("--learned-path-mix", type=float, default=1.0)
    parser.add_argument("--disable-path-residual-x0-time-rho", action="store_true",
                        help="Use rho=1 for residual x0_hat sampling instead of rho=clamp(1-t, 0, 1).")
    parser.add_argument("--x0-hat-rho-scale", type=float, default=1.0)
    parser.add_argument("--num-time-samples", type=int, default=1)
    parser.add_argument("--reset-opt-on-resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--autosave-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-num-batches", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--sanity-tol", type=float, default=5e-5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for training/eval forwards. fp16 also enables GradScaler.",
    )
    parser.add_argument("--attn-func", type=str, default=None,
                        choices=["base", "fa2", "fa3", "torch_sdpa"],
                        help="Attention backend. Default: timm built-in.")
    args = parser.parse_args()
    if args.teacher_model is None:
        args.teacher_model = args.model

    if not (0.0 <= args.beta <= 1.0):
        raise ValueError("--beta must be in [0, 1].")
    if args.fd_step <= 0.0 or args.fd_step >= 0.5:
        raise ValueError("--fd-step must lie in (0, 0.5).")
    if args.path_boundary_envelope_lambda <= 0.0:
        raise ValueError("--path-boundary-envelope-lambda must be positive.")
    if args.accel_reg_weight < 0.0:
        raise ValueError("--accel-reg-weight must be non-negative.")
    if not np.isfinite(args.path_net_ema) or args.path_net_ema < 0.0 or args.path_net_ema > 1.0:
        raise ValueError("--path_net_ema must be in [0, 1].")
    if args.feature_t_thresh < 0.0 or args.feature_t_thresh > 1.0:
        raise ValueError("--feature-t-thresh must be in [0, 1].")
    if args.feature_energy_scale < 0.0 or args.feature_energy_scale > 1.0:
        raise ValueError("--feature-energy-scale must be in [0, 1].")
    if args.feature_global_scale <= 0.0:
        raise ValueError("--feature-global-scale must be positive.")
    if args.accel_reg_weight > 0.0 and args.fd_step >= (1.0 / 3.0):
        raise ValueError("--fd-step must lie in (0, 1/3) when --accel-reg-weight is enabled.")
    if args.learned_path_mix < 0.0 or args.learned_path_mix > 1.0:
        raise ValueError("--learned-path-mix must be in [0, 1].")
    if args.autosave_every < 0:
        raise ValueError("--autosave-every must be non-negative.")
    if args.per_gpu_batch_size is not None and args.per_gpu_batch_size < 1:
        raise ValueError("--per-gpu-batch-size must be at least 1.")
    if args.num_time_samples < 1:
        raise ValueError("--num-time-samples must be at least 1.")
    if args.eval_num_batches < 1:
        raise ValueError("--eval-num-batches must be at least 1.")

    main(args)
