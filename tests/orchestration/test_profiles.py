"""Tests for the machine profiles, the platform choice and the make targets."""

import ast
import importlib
import importlib.util
import inspect
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest
import yaml

from wut_murl_craftax.orchestration import runner, runtime
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


def test_only_the_cuda_profile_has_an_unmeasured_recompute_bound() -> None:
    """ctrl-metrics review: measured on CPU and Metal, not yet on CUDA (TF32);
    ``scripts/cluster/check_recompute.py`` measures it, then cuda.yaml drops it."""
    cuda = load_profile(PROFILES / "cuda.yaml")
    assert cuda.unmeasured_limits == ("ppo/ratio_first_mb",)
    assert load_profile(PROFILES / "laptop.yaml").unmeasured_limits == ()


def test_the_cuda_group_gives_mlflow_gpu_metrics_on_linux_only() -> None:
    """MLflow's GPU monitor imports ``pynvml`` (nvidia-ml-py), else logs no GPU."""
    from mlflow.system_metrics.metrics import gpu_monitor

    assert "import pynvml" in inspect.getsource(gpu_monitor)
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    linux = "sys_platform == 'linux'"
    wanted = [r for r in pyproject["dependency-groups"]["cuda"] if "nvidia-ml-py" in r]
    assert len(wanted) == 1
    assert linux in wanted[0]
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    (project,) = (p for p in lock["package"] if p["name"] == "wut-murl-craftax")
    pin = {"name": "nvidia-ml-py", "marker": linux}
    assert pin in project["dev-dependencies"]["cuda"]
    if sys.platform != "linux":  # the laptop never gets it
        assert importlib.util.find_spec("pynvml") is None


def test_only_the_cuda_profile_drops_state_on_finish() -> None:
    """1B storage: a finished cluster run's 0.45-1.8 GB ``state/`` is never synced."""
    assert load_profile(PROFILES / "cuda.yaml").drop_state_on_finish is True
    assert load_profile(PROFILES / "laptop.yaml").drop_state_on_finish is False


def test_profiles_hold_runtime_settings_only() -> None:
    keys = {"host_devices", "device_roles", "memory_limit_bytes", "checkpoint_every"}
    keys |= {"unmeasured_limits", "drop_state_on_finish"}
    for path in PROFILES.glob("*.yaml"):
        assert yaml.safe_load(path.read_text()).keys() == keys, path.name


def test_both_profiles_resolve_the_same_run_names() -> None:
    def names(profile: str) -> str:
        # dyna-q expands only once smdp-skills has a tuned block
        args = ["--phase", "configs/phases/smoke.yaml", "--method", "sil", "--dry-run"]
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
    "unmeasured_limits": [],
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
        ("unmeasured_limits", "ppo/ratio_first_mb", "unmeasured_limits"),
        ("unmeasured_limits", [1], "unmeasured_limits"),
    ],
)
def test_an_invalid_value_fails_loudly(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_profile(_write(tmp_path, **{key: value}))


def test_a_valid_profile_round_trips(tmp_path: Path) -> None:
    assert load_profile(_write(tmp_path)) == Profile(**GOOD)  # type: ignore[arg-type]


def test_drop_state_on_finish_is_off_unless_set_and_a_boolean(tmp_path: Path) -> None:
    assert load_profile(_write(tmp_path)).drop_state_on_finish is False
    assert load_profile(
        _write(tmp_path, drop_state_on_finish=True)
    ).drop_state_on_finish
    with pytest.raises(TypeError, match="drop_state_on_finish"):
        load_profile(_write(tmp_path, drop_state_on_finish="yes"))


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


@pytest.mark.parametrize(
    ("mode", "platforms", "fails"),
    [
        ("gpu", ["cpu"], True),
        ("gpu", ["METAL", "cpu"], True),
        ("gpu", ["gpu", "cpu"], False),
        ("cpu-actor", ["cpu"], False),
    ],
)
def test_the_gpu_mode_fails_loudly_without_a_gpu_backend(
    monkeypatch: pytest.MonkeyPatch, mode: str, platforms: list[str], fails: bool
) -> None:
    """Review M11: a node whose CUDA plugin failed would train on one CPU device."""
    devices = [SimpleNamespace(platform=p) for p in platforms]
    monkeypatch.setattr(runtime.jax, "devices", lambda *_: devices)
    if fails:
        with pytest.raises(runtime.BackendError, match="no GPU backend"):
            runtime.check_backend(mode)  # type: ignore[arg-type]
    else:
        runtime.check_backend(mode)  # type: ignore[arg-type]


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


# --- cluster array script -------------------------------------------------------

ARRAY = REPO / "scripts" / "cluster" / "run_array.sh"
_ARRAY_ENV = ("SLURM_ARRAY_", "ARRAY_", "SEED_FROM", "STORE_DIR", "PROFILE")


def _array(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the array script with a fake ``uv`` that prints its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "uv").write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    (bin_dir / "uv").chmod(0o755)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(_ARRAY_ENV)}
    clean["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    return subprocess.run(
        ["bash", str(ARRAY), *args],
        cwd=tmp_path,
        env=clean | env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_array_gives_each_task_a_worker_and_its_own_store(tmp_path: Path) -> None:
    """infra-6 review M8."""
    out = _array(
        tmp_path,
        "final-1b",
        "--method",
        "ppo",
        SLURM_ARRAY_TASK_ID="2",
        SLURM_ARRAY_TASK_COUNT="4",
        SEED_FROM="/main/mlruns.db",
    )
    assert out.returncode == 0, out.stderr
    store = (tmp_path / "node-stores").resolve() / "mlruns-node-2.db"
    assert out.stdout.splitlines() == [
        *shlex.split(QUEUE)[1:],
        "--phase",
        "configs/phases/final-1b.yaml",
        "--profile",
        "configs/profiles/cuda.yaml",
        "--worker-index",
        "2",
        "--num-workers",
        "4",
        "--tracking-uri",
        f"sqlite:///{store}",
        "--node",
        "--seed-from",
        "/main/mlruns.db",
        "--method",
        "ppo",
    ]


def test_run_array_takes_generic_variables_and_needs_an_index(tmp_path: Path) -> None:
    out = _array(
        tmp_path,
        "smoke",
        ARRAY_INDEX="0",
        ARRAY_SIZE="1",
        PROFILE="laptop",
        STORE_DIR=str(tmp_path / "s"),
    )
    assert out.returncode == 0, out.stderr
    args = out.stdout.splitlines()
    assert args[args.index("--profile") + 1] == "configs/profiles/laptop.yaml"
    assert args[args.index("--tracking-uri") + 1].endswith("/s/mlruns-node-0.db")
    assert "--seed-from" not in args
    missing = _array(tmp_path, "smoke")
    assert missing.returncode != 0
    assert "no array index" in missing.stderr


def test_run_array_asks_for_sigterm_well_before_the_time_limit() -> None:
    """Review M12: the queue gets time to end its chunk and save ``state/``."""
    grace = re.search(
        r"^#SBATCH --signal=B:TERM@(\d+)$", ARRAY.read_text(), re.MULTILINE
    )
    assert grace is not None
    assert int(grace[1]) >= 300


def _script(name: str) -> Any:
    """A module of ``scripts/cluster`` (not a package), loaded from its file."""
    spec = importlib.util.spec_from_file_location(name, ARRAY.parent / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_recompute_reports_the_largest_gaps_and_the_first_ratio() -> None:
    """ctrl-metrics review: the cluster check's arithmetic, without Craftax."""
    check = _script("check_recompute")
    gap = jnp.array([[[1e-6], [-2e-6]], [[3e-6], [0.0]]])  # (minibatch, step, env)
    err = jnp.array([[[0.0], [1e-6]], [[-4e-6], [2e-6]]])
    mbs = {
        "obs": jnp.zeros((2, 2, 1, 2)),  # the toy network's logits: log-probs log 0.5
        "act": jnp.zeros((2, 2, 1), jnp.int32),
        "logp": jnp.log(0.5) - gap,
        "v": 0.5 + err,
        "h0": jnp.zeros((2, 1, 3)),
        "reset": jnp.zeros((2, 2, 1), bool),
    }

    def apply(p: Any, h0: Any, obs: Any, resets: Any) -> Any:
        return h0, obs, jnp.full(obs.shape[:-1], p)

    logp, v, ratio = check.recompute_gaps(apply, 0.5, mbs)
    assert logp == pytest.approx(3e-6, abs=1e-7)
    assert v == pytest.approx(4e-6, abs=1e-7)
    assert ratio == pytest.approx((math.exp(1e-6) + math.exp(-2e-6)) / 2, abs=1e-7)


def test_check_recompute_imports_resolve_and_its_help_runs() -> None:
    """Its imports run only on a node, so a renamed one would fail there first."""
    path = ARRAY.parent / "check_recompute.py"
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("wut_"):
            module = importlib.import_module(str(node.module))
            missing = [a.name for a in node.names if not hasattr(module, a.name)]
            assert missing == [], node.module
    out = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--precision" in out.stdout
