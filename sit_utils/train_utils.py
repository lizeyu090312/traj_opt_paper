import math
from contextlib import nullcontext

import torch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")


def resolve_amp_dtype(amp_dtype):
    if amp_dtype in (None, "none"):
        return None
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def maybe_autocast(device_type, amp_dtype):
    resolved_dtype = resolve_amp_dtype(amp_dtype)
    if resolved_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=resolved_dtype)


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------

def make_lr_schedule(
    *,
    schedule,
    base_lr,
    min_lr=0.0,
    warmup_steps=0,
    anneal_steps=None,
    start_step=0,
):
    if schedule == "none":
        return None

    if anneal_steps is None or anneal_steps <= 0:
        raise ValueError("anneal_steps must be a positive integer when an LR schedule is enabled.")
    if base_lr <= 0.0:
        raise ValueError("base_lr must be positive when an LR schedule is enabled.")
    if min_lr < 0.0:
        raise ValueError("min_lr must be non-negative.")
    if min_lr > base_lr:
        raise ValueError("min_lr must be <= base_lr.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    decay_steps = max(anneal_steps - warmup_steps, 1)

    def lr_at(global_step):
        local_step = max(int(global_step) - int(start_step), 0)

        if warmup_steps > 0 and local_step < warmup_steps:
            warmup_progress = float(local_step + 1) / float(warmup_steps)
            return base_lr * warmup_progress

        decay_step = min(max(local_step - warmup_steps, 0), decay_steps)
        decay_progress = float(decay_step) / float(decay_steps)

        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr + (base_lr - min_lr) * cosine

        raise ValueError(f"Unsupported LR schedule: {schedule}")

    return lr_at


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_optimizer_lr(optimizer):
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no parameter groups.")
    return optimizer.param_groups[0]["lr"]
