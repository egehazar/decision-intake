"""pyproject.toml is read by pip and by Vercel's Python builder, which rejects an invalid [project] table and
fails the deployment (a misplaced [project.urls] once did). Check the shapes they validate."""

import importlib
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_project_table_has_the_right_shapes():
    p = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert isinstance(p["name"], str) and isinstance(p["version"], str)
    assert isinstance(p["dependencies"], list) and all(isinstance(d, str) for d in p["dependencies"])
    assert all(isinstance(v, str) for v in p["urls"].values())
    assert all(isinstance(v, list) for v in p["optional-dependencies"].values())
    assert isinstance(p["license"], str) and isinstance(p["readme"], str)


def test_console_scripts_point_at_real_functions():
    p = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    for target in p["scripts"].values():
        module, func = target.split(":")
        assert callable(getattr(importlib.import_module(module), func))
