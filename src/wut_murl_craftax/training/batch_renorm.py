# Ported from purejaxql utils/batch_renorm.py (https://github.com/mttga/purejaxql,
# commit 47af6d7), Copyright 2024 Matteo Gallici, itself adapted from Flax's BatchNorm
# (https://github.com/google/flax, flax/linen/normalization.py), Copyright 2024 The
# Flax Authors. Both are licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License. You
# may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
# Changes: type annotations, docstrings and comments; the behaviour is the original's.
"""BatchRenorm (Ioffe 2017) as purejaxql ships it: the PQN-RNN input normalisation.

Batch statistics in training mode, the running averages otherwise. After 1,000
training calls (the original comment says 100,000; the code uses 1,000) the batch
statistics are corrected towards the running ones by r in [1/r_max, r_max] and d in
[-d_max, d_max]. ``r_max``, ``d_max`` and the call count live in ``batch_stats``.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn import initializers

WARM_UP = 1000
"""Training calls before r and d are applied."""


class BatchRenorm(Module):
    """BatchRenorm over every axis but ``axis``; momentum 0.999, epsilon 1e-3."""

    use_running_average: bool | None = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Any = None
    param_dtype: Any = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[..., jax.Array] = initializers.zeros
    scale_init: Callable[..., jax.Array] = initializers.ones
    axis_name: str | None = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(
        self, x: jax.Array, use_running_average: bool | None = None
    ) -> jax.Array:
        """Normalise ``x``; in training mode also update the running statistics."""
        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape: Sequence[int] = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats", "mean", lambda s: jnp.zeros(s, jnp.float32), feature_shape
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )
        r_max = self.variable("batch_stats", "r_max", lambda s: s, 3)
        d_max = self.variable("batch_stats", "d_max", lambda s: s, 5)
        steps = self.variable("batch_stats", "steps", lambda s: s, 0)

        if use_running_average:
            custom_mean, custom_var = ra_mean.value, ra_var.value
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean, custom_var = mean, var
            if not self.is_initializing():
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                warmed_up = jnp.greater_equal(steps.value, WARM_UP).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1.0 - warmed_up) * custom_mean

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        out = _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
        return cast(jax.Array, out)
