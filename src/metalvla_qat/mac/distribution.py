"""Shared Bernoulli parameterization for sampling and differentiable replay."""

import math

SAMPLING_VERSION = 2


def gripper_logits(action, noise_scale, xp):
    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError({"noise_scale": noise_scale})
    # actions() exposes tanh(raw); restore raw before applying temperature.
    # Clip only at representational saturation so gradients remain finite.
    return xp.arctanh(xp.clip(action, -1 + 1e-6, 1 - 1e-6)) / noise_scale
