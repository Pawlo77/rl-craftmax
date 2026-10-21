"""Machine profiles: runtime settings of a laptop or cluster node, no hyperparameters.

A run trains identically under every profile (``configs/profiles/<name>.yaml``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


def load_profile(path: Path | str) -> Profile:
    """Read a profile YAML file; a missing or unknown key raises ``TypeError``."""
    return Profile(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
