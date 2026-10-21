# Portions adapted from purejaxql pqn_rnn_craftax.py (https://github.com/mttga/purejaxql,
# commit 47af6d7), Copyright 2024 Matteo Gallici. Licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or
# agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and
# limitations under the License. Changes: the Q(lambda) targets are one reverse scan
# with the next return as the only carry; see the docstrings for the rest.
"""Recurrent PQN (LSTM) on Craftax(-Classic)-Symbolic: the PQN-RNN control and base.

Online Q(lambda) on the batch just collected, without replay or target network
(Gallici 2025; purejaxql ``pqn_rnn_craftax.py``). Design: [[PQN-RNN design]].
"""

from typing import Any, cast

import jax


def q_lambda_targets(
    q_next_max: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float | jax.Array,
    lam: float | jax.Array,
) -> jax.Array:
    """Q(lambda) targets of steps 0..T-2 of a ``(T, envs)`` rollout.

    Every input is aligned on steps t = 0..T-2: ``q_next_max[t]`` is max_a Q(s_{t+1},
    a) (stop-gradient Q of the same forward pass), ``reward[t]`` and ``done[t]`` are
    step t's. ``G_t = r_t + gamma (1 - d_t) ((1 - lam) maxQ(s_{t+1}) + lam G_{t+1})``,
    seeded with ``G_{T-1} = maxQ(s_{T-1})``, so the last target bootstraps from the
    last state of the window and never from one past it (purejaxql issue #12).
    """

    def back(
        g_next: jax.Array, x: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        q, r, d = x
        g = r + gamma * (1 - d) * ((1 - lam) * q + lam * g_next)
        return g, g

    targets: Any
    _, targets = jax.lax.scan(
        back, q_next_max[-1], (q_next_max, reward, done), reverse=True
    )
    return cast(jax.Array, targets)
