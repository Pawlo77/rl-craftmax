"""Temporary copies of configs/methods with some methods' ``tuned`` block replaced."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

METHODS = Path(__file__).resolve().parents[2] / "configs" / "methods"


def copy_methods(out: Path, tuned: Mapping[str, Mapping[str, Any]]) -> Path:
    """Copy every method file to ``out``, setting ``tuned`` of the slugs in ``tuned``.

    The key is replaced in the parsed file, so the copy does not depend on how the
    shipped file spells its block (``null``, ``{}`` or a marked config).
    """
    for path in METHODS.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if path.stem in tuned:
            raw = yaml.safe_load(text)
            assert "tuned" in raw, f"{path.name} has no tuned key"
            raw["tuned"] = dict(tuned[path.stem])
            text = yaml.safe_dump(raw, sort_keys=False)
        (out / path.name).write_text(text, encoding="utf-8")
    assert {p.stem for p in out.glob("*.yaml")} >= tuned.keys()
    return out
