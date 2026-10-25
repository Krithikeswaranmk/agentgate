"""The documentation's code runs, and its links point somewhere."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def run_in_module(code: str, name: str, module=None):
    """Execute documentation code as a real module (dataclasses need one)."""
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    exec(compile(code, name, "exec"), module.__dict__)
    return module


@pytest.fixture
def doc_module():
    created = []

    def make(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        created.append(name)
        return module

    yield make
    for name in created:
        sys.modules.pop(name, None)


def test_readme_example_runs(doc_module, capsys):
    run_in_module(python_blocks(ROOT / "README.md")[0], "readme_example", doc_module("readme_example"))
    assert "executed" in capsys.readouterr().out


def test_getting_started_examples_run(tmp_path, monkeypatch, capsys, doc_module):
    """Runs the guide's self-contained blocks offline, in order, in one module."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "report.txt").write_text("hello", encoding="utf-8")
    module = doc_module("getting_started_guide")
    ran = 0
    for block in python_blocks(ROOT / "docs" / "getting-started.md"):
        if "<model name>" in block or "config=..." in block or "# ... register" in block:
            continue  # needs an API key, or is a fragment of the reader's own setup
        run_in_module(block, "getting_started_guide", module)
        ran += 1
    assert ran >= 4
    assert "hello" in capsys.readouterr().out


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_relative_links_exist(path):
    for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
        assert (path.parent / target).exists(), f"{path.name} links to missing {target}"


def test_documented_public_names_exist():
    import saw

    reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    for name in saw.__all__:
        if name.startswith("__"):
            continue
        assert f"`{name}" in reference or name in reference, f"{name} is not in docs/reference.md"
