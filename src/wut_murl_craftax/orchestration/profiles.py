"""Machine profiles: runtime settings of a laptop or cluster node, no hyperparameters.

A run trains identically under every profile (``configs/profiles/<name>.yaml``).
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, get_args

import yaml

RoleMode = Literal["cpu-actor", "gpu"]
"""``cpu-actor``: rollouts on CPU shards, updates on the GPU when present (Metal);
``gpu``: rollouts and updates on the GPU (CUDA)."""


@dataclass(frozen=True, slots=True)
class Profile:
    """Runtime settings the runner reads once per process."""

    host_devices: int
    """CPU devices XLA splits the host into (``configure_runtime``)."""
    device_roles: RoleMode
    """Where rollouts and updates run."""
    memory_limit_bytes: int
    """Memory a method's buffers may take on this machine."""
    checkpoint_every: int
    """Most updates between full-state saves under ``state/``; results ignore it."""
    unmeasured_limits: tuple[str, ...]
    """``abort.limits`` metrics whose bound is not yet measured on this machine's
    devices: a hit is retried like a non-finite loss instead of ending the run for
    good (``scripts/cluster/check_recompute.py`` measures PPO-RNN's)."""

    def __post_init__(self) -> None:
        """Reject a value a machine cannot run with, naming the field."""
        if self.device_roles not in get_args(RoleMode):
            modes = get_args(RoleMode)
            msg = f"device_roles is one of {modes}, got {self.device_roles!r}"
            raise ValueError(msg)
        names = self.unmeasured_limits
        if not isinstance(names, list | tuple) or not all(
            isinstance(n, str) for n in names
        ):
            msg = f"unmeasured_limits is a list of metric names, got {names!r}"
            raise ValueError(msg)
        object.__setattr__(self, "unmeasured_limits", tuple(names))  # frozen, hashable
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name not in ("device_roles", "unmeasured_limits") and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                msg = f"{field.name} is a positive integer, got {value!r}"
                raise ValueError(msg)


def load_profile(path: Path | str) -> Profile:
    """Read a profile YAML file.

    Raises:
        TypeError: A key is missing or unknown.
        ValueError: A value is out of range (see `Profile`).
    """
    return Profile(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
