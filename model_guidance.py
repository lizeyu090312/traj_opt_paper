"""Utilities shared by flow and path training for Model Guidance (MG).

The functions in this module are intentionally opt-in. Callers that do not
enable MG keep their existing label handling and loss computation unchanged.
"""

import math

import torch


def validate_model_guidance_config(
    *,
    weight_low,
    weight_high,
    drop_fraction,
    data_side_threshold,
):
    if not math.isfinite(weight_low) or not math.isfinite(weight_high):
        raise ValueError("MG weights must be finite.")
    if weight_low > weight_high:
        raise ValueError("MG weight range must satisfy low <= high.")
    if not 0.0 <= drop_fraction <= 1.0:
        raise ValueError("MG drop fraction must be in [0, 1].")
    if not 0.0 <= data_side_threshold <= 1.0:
        raise ValueError("MG data-side threshold must be in [0, 1].")


def prepare_model_guidance_batch(
    labels,
    *,
    num_classes,
    weight_low,
    weight_high,
    drop_fraction,
    guidance_active,
    dtype,
):
    """Prepare MG labels, per-sample weights, and the guided prefix size.

    When MG is configured but has not reached its start step,
    ``guidance_active`` is false while deterministic label dropping remains
    active. This matches the reference implementation and lets the EMA learn
    the unconditional embedding before guided targets are introduced.
    """

    batch_size = len(labels)
    num_drop = round(drop_fraction * batch_size)
    num_guided = batch_size - num_drop if guidance_active else 0

    weights = torch.ones(batch_size, device=labels.device, dtype=dtype)
    if num_guided > 0:
        weights[:num_guided] = (
            torch.rand(num_guided, device=labels.device, dtype=dtype)
            * (weight_high - weight_low)
            + weight_low
        )

    prepared_labels = labels.clone()
    if num_drop > 0:
        drop_slice = slice(num_guided, num_guided + num_drop)
        prepared_labels[drop_slice] = num_classes
        weights[drop_slice] = 0

    return prepared_labels, weights, num_guided


def model_guidance_reference_labels(labels, *, num_classes, contrastive=False):
    if not contrastive:
        return torch.full_like(labels, num_classes)
    return (
        labels
        + torch.randint(1, num_classes, labels.shape, device=labels.device)
    ) % num_classes


def apply_model_guidance_correction(
    target,
    conditional_prediction,
    reference_prediction,
    times,
    weights,
    *,
    data_side_threshold,
):
    """Add the MG correction to the guided prefix of a velocity target."""

    num_guided = len(conditional_prediction)
    if num_guided == 0:
        return target

    data_side = times[:num_guided] > (1.0 - data_side_threshold)
    correction_weight = torch.where(
        data_side,
        weights[:num_guided] - 1.0,
        torch.zeros_like(weights[:num_guided]),
    )
    corrected = target.clone()
    corrected[:num_guided] = (
        target[:num_guided]
        + correction_weight.view(-1, *([1] * (target.ndim - 1)))
        * (conditional_prediction - reference_prediction)
    )
    return corrected
