"""Tests for the machine profiles, the platform choice and the make targets."""

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wut_murl_craftax.orchestration import runner
from wut_murl_craftax.orchestration.profiles import Profile, load_profile

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "configs" / "profiles"
GIB = 1024**3
QUEUE = "uv run python -m wut_murl_craftax.orchestration.queue"
APPLE_GPU = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and (REPO / ".venv").exists()
)


def _python(code: str, **env: str | None) -> str:
    """Run ``code`` in a fresh interpreter; ``None`` removes a variable."""
    merged = {k: v for k, v in {**os.environ, **env}.items() if v is not None}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=merged,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return result.stdout.strip().splitlines()[-1]


def _make(*args: str) -> str:
    """The command lines ``make -n`` would run."""
    result = subprocess.run(
        ["make", "-n", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --- the shipped profiles -------------------------------------------------------


def test_laptop_profile_splits_the_cpu_and_updates_on_the_gpu() -> None:
    laptop = load_profile(PROFILES / "laptop.yaml")
    assert (laptop.host_devices, laptop.device_roles) == (8, "cpu-actor")
    assert 0 < laptop.memory_limit_bytes <= 48 * GIB  # a 48 GB machine
    assert laptop.checkpoint_every > 0


def test_cuda_profile_runs_everything_on_one_device() -> None:
    cuda = load_profile(PROFILES / "cuda.yaml")
    assert (cuda.host_devices, cuda.device_roles) == (1, "gpu")
    assert cuda.memory_limit_bytes > 0
    assert cuda.checkpoint_every > 0


def test_profiles_hold_runtime_settings_only() -> None:
    keys = {"host_devices", "device_roles", "memory_limit_bytes", "checkpoint_every"}
    for path in PROFILES.glob("*.yaml"):
        assert yaml.safe_load(path.read_text()).keys() == keys, path.name


def test_both_profiles_resolve_the_same_run_names() -> None:
    def names(profile: str) -> str:
        args = ["--phase", "configs/phases/smoke.yaml", "--dry-run"]
        args += ["--profile", str(PROFILES / f"{profile}.yaml")]
        cmd = [sys.executable, "-m", "wut_murl_craftax.orchestration.queue", *args]
        return subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, check=True
        ).stdout

    assert names("laptop") == names("cuda") != ""


# --- validation -----------------------------------------------------------------

GOOD = {
    "host_devices": 8,
    "device_roles": "cpu-actor",
    "memory_limit_bytes": GIB,
    "checkpoint_every": 10,
}


def _write(tmp_path: Path, **changes: object) -> Path:
    body = {**GOOD, **changes}
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in body.items() if v is not None}))
    return path


@pytest.mark.parametrize("key", sorted(GOOD))
def test_a_missing_key_fails_loudly(tmp_path: Path, key: str) -> None:
    with pytest.raises(TypeError, match=key):
        load_profile(_write(tmp_path, **{key: None}))


def test_an_unknown_key_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="learning_rate"):
        load_profile(_write(tmp_path, learning_rate=1e-3))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("device_roles", "tpu", "device_roles"),
        ("host_devices", 0, "host_devices"),
        ("memory_limit_bytes", -1, "memory_limit_bytes"),
        ("checkpoint_every", 0, "checkpoint_every"),
        ("checkpoint_every", 2.5, "checkpoint_every"),
        ("host_devices", True, "host_devices"),
    ],
)
def test_an_invalid_value_fails_loudly(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_profile(_write(tmp_path, **{key: value}))


def test_a_valid_profile_round_trips(tmp_path: Path) -> None:
    assert load_profile(_write(tmp_path)) == Profile(**GOOD)  # type: ignore[arg-type]


# --- platform choice ------------------------------------------------------------

_SELECT = """
import os
from wut_murl_craftax.orchestration import runtime
runtime.metal_installed = lambda: {installed}
runtime.select_platforms("{mode}")
import jax
print(os.environ.get("JAX_PLATFORMS"), jax.config.jax_platforms)
"""


@pytest.mark.parametrize(
    ("installed", "mode", "explicit", "want"),
    [
        (True, "cpu-actor", None, "metal,cpu metal,cpu"),
        (False, "cpu-actor", None, "None None"),
        (True, "gpu", None, "None None"),
        (True, "cpu-actor", "cpu", "cpu cpu"),
    ],
    ids=["metal", "no-metal", "cuda-profile", "explicit-wins"],
)
def test_select_platforms(
    installed: bool, mode: str, explicit: str | None, want: str
) -> None:
    code = _SELECT.format(installed=installed, mode=mode)
    assert _python(code, JAX_PLATFORMS=explicit) == want


@pytest.mark.skipif(not APPLE_GPU, reason="needs Apple Silicon with metaljax")
def test_laptop_profile_gives_a_metal_learner_and_cpu_shards() -> None:
    code = """
import jax
from wut_murl_craftax.orchestration import runtime
from wut_murl_craftax.orchestration.profiles import load_profile
if not runtime.metal_installed():
    print("skip")
else:
    profile = load_profile("configs/profiles/laptop.yaml")
    runtime.select_platforms(profile.device_roles)
    runtime.configure_runtime(host_devices=profile.host_devices)
    roles = runtime.device_roles()
    print(roles.learner.platform, roles.actor.platform, len(jax.devices("cpu")))
"""
    out = _python(code, JAX_PLATFORMS=None, XLA_FLAGS="")
    if out == "skip":
        pytest.skip("metaljax is not installed")
    assert out == "metal cpu 8"


def test_the_runner_picks_the_platform_before_it_splits_the_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where runs train (each child, `_set_up_jax`): platforms, CPU split, Craftax,
    then the registry; the queue parent itself starts no JAX backend."""
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        runner, "select_platforms", lambda mode: calls.append(("platforms", mode))
    )
    monkeypatch.setattr(
        runner, "configure_runtime", lambda **kw: calls.append(("runtime", kw))
    )
    monkeypatch.setattr(runner, "import_craftax", lambda: calls.append(("craftax", 0)))
    monkeypatch.setattr(
        runner, "_imported", lambda ref: calls.append(("registry", ref)) or {}
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump({
            "experiment": {"name": "craftax"},
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_path / 'mlruns.db'}",
                "experiment_name": "x",
            },
        })
    )
    runner.main([], PROFILES / "laptop.yaml", experiment_config=experiment)
    assert calls == []  # the parent: no runs here, children set JAX up
    runner._set_up_jax(load_profile(PROFILES / "laptop.yaml"), runner.REGISTRY, None)
    assert calls == [
        ("platforms", "cpu-actor"),
        ("runtime", {"host_devices": 8}),
        ("craftax", 0),
        ("registry", runner.REGISTRY),
    ]


# --- make targets ---------------------------------------------------------------


def test_make_queue_calls_the_queue_cli() -> None:
    out = _make("queue", "PHASE=tune-coarse", "METHOD=count-bonus")
    assert shlex.split(out) == shlex.split(
        f"{QUEUE} --phase configs/phases/tune-coarse.yaml"
        " --profile configs/profiles/laptop.yaml --method count-bonus"
    )


def test_make_queue_fine_stage_and_dry_run() -> None:
    out = _make("queue", "PHASE=tune-fine", "METHOD=ppo", "FINE=ppo", "DRY_RUN=1")
    assert shlex.split(out) == shlex.split(
        f"{QUEUE} --phase configs/phases/tune-fine.yaml"
        " --profile configs/profiles/laptop.yaml --method ppo --expand-fine ppo"
        " --dry-run"
    )


def test_make_queue_takes_the_profile() -> None:
    out = _make("queue", "PHASE=smoke", "PROFILE=cuda")
    assert "--profile configs/profiles/cuda.yaml" in out


def test_make_queue_needs_a_phase() -> None:
    result = subprocess.run(
        ["make", "-n", "queue"], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "PHASE" in result.stderr


STORE = "uv run python -m wut_murl_craftax.orchestration.store"


def test_make_status_calls_the_store_cli() -> None:
    out = _make("status", "PHASE=tune-coarse")
    assert shlex.split(out) == shlex.split(
        f"{STORE} status --phase configs/phases/tune-coarse.yaml"
    )


def test_make_status_takes_method_and_fine() -> None:
    out = _make("status", "PHASE=tune-fine", "METHOD=ppo", "FINE=ppo")
    assert out.endswith("--method ppo --expand-fine ppo")


def test_make_merge_mlflow_calls_the_store_cli() -> None:
    out = _make("merge-mlflow", "SRC=mlruns-node-3.db")
    assert shlex.split(out) == shlex.split(f"{STORE} merge mlruns-node-3.db")


def test_make_merge_mlflow_needs_a_source() -> None:
    result = subprocess.run(
        ["make", "-n", "merge-mlflow"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SRC" in result.stderr
