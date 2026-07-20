import torch, torch.nn.functional as F

from model_guidance import (
    apply_model_guidance_correction,
    prepare_model_guidance_batch,
    validate_model_guidance_config,
)


class VAEDecoderAdapter:
    """
    Decodes latent z_t into image space for feature-JVP computation.
    The adapter is callable to make decoder swaps straightforward. Only use sd-vae-mse/sd-vae-ema
    """
    def __init__(self, vae):
        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
    def __call__(self, z_t):
        return self.vae.decode(z_t / 0.18215).sample

class FeaturePhi:
    # Expects input x in [-1, 1], shape [N, 3, H, W].
    def __init__(self, model_name, device):
        self.model_name = model_name
        if model_name == "InceptionV3":
            from pytorch_fid.inception import InceptionV3

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.m = InceptionV3([block_idx], resize_input=True, normalize_input=False).to(device).eval()
            for p in self.m.parameters():
                p.requires_grad_(False)
            return

        import timm
        import torchvision

        self.m = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.img_size = cfg["input_size"][-2:]  # (H, W)
        self.norm = torchvision.transforms.Normalize(mean=cfg["mean"], std=cfg["std"])

    def _prep_inception(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * 0.5) + 0.5
        x = F.interpolate(x, size=self.img_size, mode="bicubic", align_corners=False)
        return self.norm(x)

    def __call__(self, x):
        if self.model_name == "InceptionV3":
            return self.m(self._prep_inception(x))[0].flatten(1)
        return self.m(self._prep(x))


def disable_label_dropout(model):
    if hasattr(model, "y_embedder") and hasattr(model.y_embedder, "dropout_prob"):
        model.y_embedder.dropout_prob = 0.0


def zero_sit_output_layer(model):
    with torch.no_grad():
        model.final_layer.linear.weight.zero_()
        model.final_layer.linear.bias.zero_()
        model.final_layer.adaLN_modulation[-1].weight.zero_()
        model.final_layer.adaLN_modulation[-1].bias.zero_()


def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False


def _expand_t_like_x(t, x):
    return t.view(-1, *([1] * (x.ndim - 1)))


def sample_path_residual_x0_hat(x0, t, *, disable_path_residual_x0_time_rho=False, path_rho_constant=None, x0_hat_rho_scale=1.0):
    if path_rho_constant is None and disable_path_residual_x0_time_rho:
        return x0
    if torch.is_tensor(t):
        t = t.to(device=x0.device, dtype=x0.dtype)
    else:
        t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
    if path_rho_constant is None:
        rho = torch.clamp(x0_hat_rho_scale * (1.0 - t), 0.0, 1.0)
    else:
        rho = torch.full_like(t, float(path_rho_constant))
    rho_view = _expand_t_like_x(rho, x0)
    noise_scale_sq = torch.clamp(1.0 - rho_view * rho_view, min=0.0)
    return rho_view * x0 + torch.sqrt(noise_scale_sq) * torch.randn_like(x0)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _fd_first_derivative(values_tp, values_tm, tp, tm):
    dt_view = _expand_t_like_x(tp - tm, values_tp)
    return (values_tp - values_tm) / dt_view


def _interpolate_values_at_t(values_tp, values_tm, t, tp, tm):
    weight_view = _expand_t_like_x((t - tm) / (tp - tm), values_tp)
    return values_tm + weight_view * (values_tp - values_tm)


def dual_stem_teacher_residual_envelope(path_model, t, x_like):
    base_model = _unwrap_model(path_model)
    boundary_lambda = float(getattr(base_model, "teacher_residual_boundary_lambda", 1.0))
    if boundary_lambda <= 0.0:
        raise ValueError(
            "dual-stem residual boundary lambda must be positive, "
            f"got {boundary_lambda}."
        )
    t_view = _expand_t_like_x(t, x_like)
    base = t_view * (1.0 - t_view)
    if boundary_lambda == 1.0:
        return base
    scale = 0.25 ** (1.0 - boundary_lambda)
    return torch.pow(base, boundary_lambda) * scale


def resolve_path_parameterization(path_model, path_parameterization=None):
    if path_parameterization is not None:
        if path_parameterization in {"legacy", "legacy_sit"}:
            return "legacy"
        if path_parameterization in {"dual_stem_teacher_residual", "dual_stem_teacher_residual_subboundary", "dual_stem_direct_residual"}:
            return path_parameterization
        raise ValueError(f"Unsupported path_parameterization: {path_parameterization}")

    base_model = _unwrap_model(path_model)
    if getattr(base_model, "uses_dual_stem_path", False) and getattr(base_model, "uses_teacher_residualized_path", False):
        return "dual_stem_teacher_residual"
    return "legacy"


def linear_path(x0, x1, t):
    t_view = _expand_t_like_x(t, x0)
    return (1.0 - t_view) * x0 + t_view * x1


def learned_path(path_model, x0, x1, y, t):
    gamma_lin = linear_path(x0, x1, t)
    residual = path_model(gamma_lin, t, y)
    t_view = _expand_t_like_x(t, gamma_lin)
    return gamma_lin + t_view * (1.0 - t_view) * residual


def _scaled_subtractor_residual(path_velocity, subtractor_velocity, subtractor_residual_scale):
    return subtractor_residual_scale * (path_velocity - subtractor_velocity)


def _resolve_path_subtractor(teacher, path_subtractor):
    if path_subtractor is not None:
        return path_subtractor
    return teacher


def dual_stem_teacher_residual_path(
    path_model,
    path_subtractor,
    x0,
    x1,
    y,
    t,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    gamma_lin = linear_path(x0, x1, t)
    if x0_hat is None:
        raise ValueError("x0_hat must be explicitly provided for dual_stem_teacher_residual paths.")
    gamma_lin_hat = linear_path(x0_hat, x1, t)
    delta_hat = x1 - x0_hat
    path_velocity = path_model(gamma_lin_hat, delta_hat, t, y)
    with torch.no_grad():
        subtractor_velocity_on_linear = path_subtractor(gamma_lin_hat, t, y)
    envelope = dual_stem_teacher_residual_envelope(path_model, t, gamma_lin)
    return gamma_lin + envelope * _scaled_subtractor_residual(
        path_velocity,
        subtractor_velocity_on_linear,
        subtractor_residual_scale,
    )


def dual_stem_direct_residual_path(path_model, x0, x1, y, t, x0_hat=None):
    gamma_lin = linear_path(x0, x1, t)
    if x0_hat is None:
        raise ValueError("x0_hat must be explicitly provided for dual_stem_direct_residual paths.")
    gamma_lin_hat = linear_path(x0_hat, x1, t)
    delta_hat = x1 - x0_hat
    path_velocity = path_model(gamma_lin_hat, delta_hat, t, y)
    envelope = dual_stem_teacher_residual_envelope(path_model, t, gamma_lin)
    return gamma_lin + envelope * path_velocity


def dual_stem_teacher_residual_correction(
    path_model,
    path_subtractor,
    x,
    delta,
    y,
    t,
    subtractor_residual_scale=1.0,
):
    path_velocity = path_model(x, delta, t, y)
    with torch.no_grad():
        subtractor_velocity = path_subtractor(x, t, y)
    return _scaled_subtractor_residual(
        path_velocity,
        subtractor_velocity,
        subtractor_residual_scale,
    )


def dual_stem_teacher_residual_subboundary_terms(
    path_model,
    path_subtractor,
    x0,
    x1,
    y,
    subtractor_residual_scale=1.0,
):
    delta = x1 - x0
    t0 = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    t1 = torch.ones(x1.shape[0], device=x1.device, dtype=x1.dtype)
    correction_0 = dual_stem_teacher_residual_correction(
        path_model,
        path_subtractor,
        x0,
        delta,
        y,
        t0,
        subtractor_residual_scale=subtractor_residual_scale,
    )
    correction_1 = dual_stem_teacher_residual_correction(
        path_model,
        path_subtractor,
        x1,
        delta,
        y,
        t1,
        subtractor_residual_scale=subtractor_residual_scale,
    )
    return delta, correction_0, correction_1


def dual_stem_teacher_residual_subboundary_path_from_terms(
    path_model,
    path_subtractor,
    x0,
    x1,
    y,
    t,
    delta,
    correction_0,
    correction_1,
    subtractor_residual_scale=1.0,
):
    gamma_lin = linear_path(x0, x1, t)
    correction_t = dual_stem_teacher_residual_correction(
        path_model,
        path_subtractor,
        gamma_lin,
        delta,
        y,
        t,
        subtractor_residual_scale=subtractor_residual_scale,
    )
    t_view = _expand_t_like_x(t, gamma_lin)
    return gamma_lin + correction_t - (1.0 - t_view) * correction_0 - t_view * correction_1


def dual_stem_teacher_residual_subboundary_path(
    path_model,
    path_subtractor,
    x0,
    x1,
    y,
    t,
    subtractor_residual_scale=1.0,
):
    delta, correction_0, correction_1 = dual_stem_teacher_residual_subboundary_terms(
        path_model,
        path_subtractor,
        x0,
        x1,
        y,
        subtractor_residual_scale=subtractor_residual_scale,
    )
    return dual_stem_teacher_residual_subboundary_path_from_terms(
        path_model,
        path_subtractor,
        x0,
        x1,
        y,
        t,
        delta,
        correction_0,
        correction_1,
        subtractor_residual_scale=subtractor_residual_scale,
    )


def generalized_learned_path(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    resolved = resolve_path_parameterization(path_model, path_parameterization=path_parameterization)
    resolved_subtractor = _resolve_path_subtractor(teacher, path_subtractor)
    if resolved == "legacy":
        return learned_path(path_model, x0, x1, y, t)
    if resolved == "dual_stem_teacher_residual":
        if resolved_subtractor is None:
            raise ValueError("path_subtractor must be provided for dual_stem_teacher_residual paths.")
        return dual_stem_teacher_residual_path(path_model, resolved_subtractor, x0, x1, y, t, subtractor_residual_scale=subtractor_residual_scale, x0_hat=x0 if x0_hat is None else x0_hat,)
    if resolved == "dual_stem_direct_residual":
        return dual_stem_direct_residual_path(path_model, x0, x1, y, t, x0_hat=x0 if x0_hat is None else x0_hat)
    if resolved == "dual_stem_teacher_residual_subboundary":
        if resolved_subtractor is None:
            raise ValueError("path_subtractor must be provided for dual_stem_teacher_residual_subboundary paths.")
        return dual_stem_teacher_residual_subboundary_path(path_model, resolved_subtractor, x0, x1, y, t, subtractor_residual_scale=subtractor_residual_scale,)
    raise ValueError(f"Unsupported path_parameterization: {resolved}")


def generalized_mixed_path(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    learned_mix=1.0,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    if learned_mix <= 0.0:
        return linear_path(x0, x1, t)
    if learned_mix >= 1.0:
        return generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            t,
            path_parameterization=path_parameterization,
            path_subtractor=path_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )

    gamma_lin = linear_path(x0, x1, t)
    gamma_learned = generalized_learned_path(
        path_model,
        teacher,
        x0,
        x1,
        y,
        t,
        path_parameterization=path_parameterization,
        path_subtractor=path_subtractor,
        subtractor_residual_scale=subtractor_residual_scale,
        x0_hat=x0_hat,
    )
    return (1.0 - learned_mix) * gamma_lin + learned_mix * gamma_learned


def generalized_path_derivative_fd(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    h,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    resolved = resolve_path_parameterization(path_model, path_parameterization=path_parameterization)
    resolved_subtractor = _resolve_path_subtractor(teacher, path_subtractor)
    tp = torch.clamp(t + h, 0.0, 1.0)
    tm = torch.clamp(t - h, 0.0, 1.0)

    if resolved == "dual_stem_teacher_residual_subboundary":
        delta, correction_0, correction_1 = dual_stem_teacher_residual_subboundary_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_t = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            t,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_tp = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            tp,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_tm = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            tm,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
    else:
        gamma_t = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            t,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_tp = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tp,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_tm = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tm,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
    gamma_dot_t = torch.empty_like(gamma_t)

    forward_mask = t < h
    backward_mask = t > (1.0 - h)
    central_mask = ~(forward_mask | backward_mask)

    if forward_mask.any():
        gamma_dot_t[forward_mask] = (gamma_tp[forward_mask] - gamma_t[forward_mask]) / h
    if backward_mask.any():
        gamma_dot_t[backward_mask] = (gamma_t[backward_mask] - gamma_tm[backward_mask]) / h
    if central_mask.any():
        gamma_dot_t[central_mask] = (gamma_tp[central_mask] - gamma_tm[central_mask]) / (2.0 * h)
    return gamma_t, gamma_dot_t


def generalized_path_geometry_fd(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    h,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    resolved = resolve_path_parameterization(path_model, path_parameterization=path_parameterization)
    resolved_subtractor = _resolve_path_subtractor(teacher, path_subtractor)
    tp = torch.clamp(t + h, 0.0, 1.0)
    tm = torch.clamp(t - h, 0.0, 1.0)

    if resolved == "dual_stem_teacher_residual_subboundary":
        delta, correction_0, correction_1 = dual_stem_teacher_residual_subboundary_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_t = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            t,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_tp = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            tp,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
        gamma_tm = dual_stem_teacher_residual_subboundary_path_from_terms(
            path_model,
            resolved_subtractor,
            x0,
            x1,
            y,
            tm,
            delta,
            correction_0,
            correction_1,
            subtractor_residual_scale=subtractor_residual_scale,
        )
    else:
        gamma_t = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            t,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_tp = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tp,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_tm = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tm,
            path_parameterization=resolved,
            path_subtractor=resolved_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
    gamma_dot_t = torch.empty_like(gamma_t)
    gamma_ddot_t = torch.empty_like(gamma_t)

    forward_mask = t < h
    backward_mask = t > (1.0 - h)
    central_mask = ~(forward_mask | backward_mask)

    if forward_mask.any():
        gamma_dot_t[forward_mask] = (gamma_tp[forward_mask] - gamma_t[forward_mask]) / h
        tpp = torch.clamp(t[forward_mask] + 2.0 * h, 0.0, 1.0)
        if resolved == "dual_stem_teacher_residual_subboundary":
            gamma_tpp = dual_stem_teacher_residual_subboundary_path_from_terms(
                path_model,
                resolved_subtractor,
                x0[forward_mask],
                x1[forward_mask],
                y[forward_mask],
                tpp,
                delta[forward_mask],
                correction_0[forward_mask],
                correction_1[forward_mask],
                subtractor_residual_scale=subtractor_residual_scale,
            )
        else:
            gamma_tpp = generalized_learned_path(
                path_model,
                teacher,
                x0[forward_mask],
                x1[forward_mask],
                y[forward_mask],
                tpp,
                path_parameterization=resolved,
                path_subtractor=resolved_subtractor,
                subtractor_residual_scale=subtractor_residual_scale,
                x0_hat=None if x0_hat is None else x0_hat[forward_mask],
            )
        gamma_ddot_t[forward_mask] = (
            gamma_tpp
            - 2.0 * gamma_tp[forward_mask]
            + gamma_t[forward_mask]
        ) / (h * h)
    if backward_mask.any():
        gamma_dot_t[backward_mask] = (gamma_t[backward_mask] - gamma_tm[backward_mask]) / h
        tmm = torch.clamp(t[backward_mask] - 2.0 * h, 0.0, 1.0)
        if resolved == "dual_stem_teacher_residual_subboundary":
            gamma_tmm = dual_stem_teacher_residual_subboundary_path_from_terms(
                path_model,
                resolved_subtractor,
                x0[backward_mask],
                x1[backward_mask],
                y[backward_mask],
                tmm,
                delta[backward_mask],
                correction_0[backward_mask],
                correction_1[backward_mask],
                subtractor_residual_scale=subtractor_residual_scale,
            )
        else:
            gamma_tmm = generalized_learned_path(
                path_model,
                teacher,
                x0[backward_mask],
                x1[backward_mask],
                y[backward_mask],
                tmm,
                path_parameterization=resolved,
                path_subtractor=resolved_subtractor,
                subtractor_residual_scale=subtractor_residual_scale,
                x0_hat=None if x0_hat is None else x0_hat[backward_mask],
            )
        gamma_ddot_t[backward_mask] = (
            gamma_t[backward_mask]
            - 2.0 * gamma_tm[backward_mask]
            + gamma_tmm
        ) / (h * h)
    if central_mask.any():
        gamma_dot_t[central_mask] = (gamma_tp[central_mask] - gamma_tm[central_mask]) / (2.0 * h)
        gamma_ddot_t[central_mask] = (
            gamma_tp[central_mask]
            - 2.0 * gamma_t[central_mask]
            + gamma_tm[central_mask]
        ) / (h * h)
    return gamma_t, gamma_dot_t, gamma_ddot_t


def generalized_mixed_path_derivative_fd(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    h,
    learned_mix=1.0,
    path_parameterization=None,
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    x0_hat=None,
):
    if learned_mix <= 0.0:
        gamma_t = linear_path(x0, x1, t)
        gamma_dot_t = x1 - x0
        return gamma_t, gamma_dot_t
    if learned_mix >= 1.0:
        return generalized_path_derivative_fd(
            path_model,
            teacher,
            x0,
            x1,
            y,
            t,
            h,
            path_parameterization=path_parameterization,
            path_subtractor=path_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )

    gamma_learned_t, gamma_learned_dot_t = generalized_path_derivative_fd(
        path_model,
        teacher,
        x0,
        x1,
        y,
        t,
        h,
        path_parameterization=path_parameterization,
        path_subtractor=path_subtractor,
        subtractor_residual_scale=subtractor_residual_scale,
        x0_hat=x0_hat,
    )
    gamma_lin_t = linear_path(x0, x1, t)
    gamma_lin_dot_t = x1 - x0
    gamma_t = (1.0 - learned_mix) * gamma_lin_t + learned_mix * gamma_learned_t
    gamma_dot_t = (1.0 - learned_mix) * gamma_lin_dot_t + learned_mix * gamma_learned_dot_t
    return gamma_t, gamma_dot_t


def teacher_track_energy(
    teacher,
    gamma_t,
    gamma_dot_t,
    y,
    t,
    *,
    mg_active=False,
    mg_w_lo=1.45,
    mg_w_hi=1.45,
    mg_drop_frac=0.1,
    mg_data_side_threshold=0.75,
    num_classes=1000,
):
    if not mg_active:
        teacher_velocity = teacher(gamma_t, t, y)
        return (gamma_dot_t - teacher_velocity).flatten(1).pow(2).mean(dim=1)

    validate_model_guidance_config(
        weight_low=mg_w_lo,
        weight_high=mg_w_hi,
        drop_fraction=mg_drop_frac,
        data_side_threshold=mg_data_side_threshold,
    )
    y_mg, weights, num_guided = prepare_model_guidance_batch(
        y,
        num_classes=num_classes,
        weight_low=mg_w_lo,
        weight_high=mg_w_hi,
        drop_fraction=mg_drop_frac,
        guidance_active=True,
        dtype=gamma_t.dtype,
    )
    teacher_velocity = teacher(gamma_t, t, y_mg)
    if num_guided > 0:
        conditional_prediction = teacher_velocity[:num_guided].detach()
        with torch.no_grad():
            uncond_y = torch.full_like(y[:num_guided], num_classes)
            reference_prediction = teacher(
                gamma_t[:num_guided],
                t[:num_guided],
                uncond_y,
            )
        teacher_velocity = apply_model_guidance_correction(
            teacher_velocity,
            conditional_prediction,
            reference_prediction,
            t,
            weights,
            data_side_threshold=mg_data_side_threshold,
        )
    return (gamma_dot_t - teacher_velocity).flatten(1).pow(2).mean(dim=1)


def euclidean_energy(gamma_dot_t):
    return gamma_dot_t.flatten(1).pow(2).mean(dim=1)


def acceleration_energy(gamma_ddot_t):
    return gamma_ddot_t.flatten(1).pow(2).mean(dim=1)


def feature_path_energy(gamma_tp, gamma_tm, t, tp, tm, feature_decoder, feature_phi, t_thresh=0.0):
    feature_tp = feature_phi(feature_decoder(gamma_tp))
    feature_tm = feature_phi(feature_decoder(gamma_tm))
    feature_dot_t = _fd_first_derivative(feature_tp, feature_tm, tp, tm)
    mask_t = (t >= t_thresh).to(feature_dot_t.dtype)
    return mask_t * euclidean_energy(feature_dot_t)


def blended_feature_energy_loss(euclid, feature_energy, feature_energy_scale, feature_global_scale):
    if feature_energy_scale < 0.0 or feature_energy_scale > 1.0:
        raise ValueError("feature_energy_scale must be in [0, 1] for feature_energy loss.")
    if feature_global_scale <= 0.0:
        raise ValueError("feature_global_scale must be positive for feature_energy loss.")
    return (1.0 - feature_energy_scale) * euclid + feature_energy_scale * feature_energy / feature_global_scale


def _metric_dict(total, track, euclid, feature_energy, accel, gamma_t, gamma_dot_t, gamma_ddot_t):
    return {
        "total": total, "track": track, "euclid": euclid, "feature_energy": feature_energy, "accel": accel, 
        "gamma_t": gamma_t, "gamma_dot_t": gamma_dot_t, "gamma_ddot_t": gamma_ddot_t,
    }


def blended_feature_energy_metrics(
    gamma_tp, gamma_tm, t, tp, tm, feature_decoder, feature_phi, feature_t_thresh=0.0, 
    feature_energy_scale=1.0, feature_global_scale=1.0,
):
    if feature_decoder is None or feature_phi is None:
        raise ValueError("feature_decoder and feature_phi must be provided for feature_energy loss.")
    gamma_t = _interpolate_values_at_t(gamma_tp, gamma_tm, t, tp, tm)
    gamma_dot_t = _fd_first_derivative(gamma_tp, gamma_tm, tp, tm)
    euclid = euclidean_energy(gamma_dot_t)
    feature_energy = feature_path_energy(gamma_tp, gamma_tm, t, tp, tm, feature_decoder, feature_phi, t_thresh=feature_t_thresh)
    zero = torch.zeros_like(euclid)
    total = blended_feature_energy_loss(euclid, feature_energy, feature_energy_scale=feature_energy_scale, feature_global_scale=feature_global_scale)
    return _metric_dict(total, zero, euclid, feature_energy, zero, gamma_t, gamma_dot_t, torch.zeros_like(gamma_t))


def combined_path_energy(
    path_model,
    teacher,
    x0,
    x1,
    y,
    t,
    beta=0.1,
    h=0.02,
    learned_mix=1.0,
    accel_reg_weight=0.0,
    path_parameterization=None,
    loss_mode="track_mixed_euclid_learned",
    path_subtractor=None,
    subtractor_residual_scale=1.0,
    feature_decoder=None,
    feature_phi=None,
    feature_t_thresh=0.0,
    feature_energy_scale=1.0,
    feature_global_scale=1.0,
    x0_hat=None,
    mg_active=False,
    mg_w_lo=1.45,
    mg_w_hi=1.45,
    mg_drop_frac=0.1,
    mg_data_side_threshold=0.75,
    num_classes=1000,
):
    if loss_mode not in {"track_mixed_euclid_learned", "feature_energy"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    if loss_mode == "feature_energy":
        tp = torch.clamp(t + h, 0.0, 1.0)
        tm = torch.clamp(t - h, 0.0, 1.0)
        if feature_decoder is None or feature_phi is None:
            raise ValueError("feature_decoder and feature_phi must be provided for feature_energy loss.")
        gamma_learned_tp = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tp,
            path_parameterization=path_parameterization,
            path_subtractor=path_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_learned_tm = generalized_learned_path(
            path_model,
            teacher,
            x0,
            x1,
            y,
            tm,
            path_parameterization=path_parameterization,
            path_subtractor=path_subtractor,
            subtractor_residual_scale=subtractor_residual_scale,
            x0_hat=x0_hat,
        )
        gamma_lin_tp = linear_path(x0, x1, tp)
        gamma_lin_tm = linear_path(x0, x1, tm)
        gamma_tp = (1.0 - learned_mix) * gamma_lin_tp + learned_mix * gamma_learned_tp
        gamma_tm = (1.0 - learned_mix) * gamma_lin_tm + learned_mix * gamma_learned_tm
        gamma_t = _interpolate_values_at_t(gamma_tp, gamma_tm, t, tp, tm)
        gamma_dot_t = _fd_first_derivative(gamma_tp, gamma_tm, tp, tm)
        gamma_ddot_t = torch.zeros_like(gamma_t)
        euclid = euclidean_energy(gamma_dot_t)
        feature_energy = feature_path_energy(
            gamma_tp,
            gamma_tm,
            t,
            tp,
            tm,
            feature_decoder,
            feature_phi,
            t_thresh=feature_t_thresh,
        )
        zero = torch.zeros_like(euclid)
        total = blended_feature_energy_loss(
            euclid,
            feature_energy,
            feature_energy_scale=feature_energy_scale,
            feature_global_scale=feature_global_scale,
        )
        return _metric_dict(total, zero, euclid, feature_energy, zero, gamma_t, gamma_dot_t, gamma_ddot_t)
    if mg_active:
        num_drop = round(mg_drop_frac * len(y))
        y_for_path = y.clone()
        if num_drop > 0:
            y_for_path[len(y) - num_drop:] = num_classes
    else:
        y_for_path = y
    gamma_learned_t, gamma_learned_dot_t, gamma_learned_ddot_t = generalized_path_geometry_fd(
        path_model, teacher, x0, x1, y_for_path, t, h, path_parameterization=path_parameterization, path_subtractor=path_subtractor, subtractor_residual_scale=subtractor_residual_scale, x0_hat=x0_hat,
    )
    gamma_lin_t = linear_path(x0, x1, t)
    gamma_lin_dot_t = x1 - x0
    gamma_t = (1.0 - learned_mix) * gamma_lin_t + learned_mix * gamma_learned_t
    gamma_dot_t = (1.0 - learned_mix) * gamma_lin_dot_t + learned_mix * gamma_learned_dot_t
    gamma_ddot_t = learned_mix * gamma_learned_ddot_t
    track = teacher_track_energy(
        teacher,
        gamma_t,
        gamma_dot_t,
        y,
        t,
        mg_active=mg_active,
        mg_w_lo=mg_w_lo,
        mg_w_hi=mg_w_hi,
        mg_drop_frac=mg_drop_frac,
        mg_data_side_threshold=mg_data_side_threshold,
        num_classes=num_classes,
    )
    euclid = euclidean_energy(gamma_learned_dot_t)
    feature_energy = torch.zeros_like(euclid)
    accel = acceleration_energy(gamma_learned_ddot_t)
    total = beta * track + (1.0 - beta) * euclid + accel_reg_weight * accel
    return _metric_dict(total, track, euclid, feature_energy, accel, gamma_t, gamma_dot_t, gamma_ddot_t)


def linear_baseline_energy(
    teacher, x0, x1, y, t, beta=0.1, h=0.02,
    loss_mode="track_mixed_euclid_learned",
    feature_decoder=None,
    feature_phi=None,
    feature_t_thresh=0.0,
    feature_energy_scale=1.0,
    feature_global_scale=1.0,
):
    if loss_mode not in {"track_mixed_euclid_learned", "feature_energy"}:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    gamma_t = linear_path(x0, x1, t)
    gamma_dot_t = x1 - x0
    gamma_ddot_t = torch.zeros_like(gamma_t)

    if loss_mode == "feature_energy":
        tp = torch.clamp(t + h, 0.0, 1.0)
        tm = torch.clamp(t - h, 0.0, 1.0)
        gamma_tp = linear_path(x0, x1, tp)
        gamma_tm = linear_path(x0, x1, tm)
        return blended_feature_energy_metrics(
            gamma_tp, gamma_tm, t, tp, tm, feature_decoder, feature_phi, feature_t_thresh=feature_t_thresh, feature_energy_scale=feature_energy_scale, feature_global_scale=feature_global_scale,
        )
    track = teacher_track_energy(teacher, gamma_t, gamma_dot_t, y, t)
    euclid = euclidean_energy(gamma_dot_t)
    feature_energy = torch.zeros_like(euclid)
    accel = acceleration_energy(gamma_ddot_t)
    total = beta * track + (1.0 - beta) * euclid
    return _metric_dict(total, track, euclid, feature_energy, accel, gamma_t, gamma_dot_t, gamma_ddot_t)
