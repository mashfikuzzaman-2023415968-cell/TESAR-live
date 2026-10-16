"""Code polish rules (spec Phase 14 item 4): every module has a docstring, every public
function has type hints, and nothing is left marked as a placeholder."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers.procs import REPO

SOURCES = sorted(
    [*REPO.glob("mdc/**/*.py"), *REPO.glob("scripts/*.py"), *REPO.glob("experiments/*.py")]
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(REPO)))
def test_module_docstring_and_typed_public_functions(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), "module docstring missing"
    untyped = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            args = [
                a for a in node.args.args + node.args.kwonlyargs if a.arg not in {"self", "cls"}
            ]
            if node.returns is None or any(a.annotation is None for a in args):
                untyped.append(f"{node.name}:{node.lineno}")
    assert not untyped, untyped


def test_no_placeholders_left() -> None:
    hits = []
    for path in [*SOURCES, *REPO.glob("docs/*.md"), *REPO.glob("configs/**/*.yaml")]:
        if "Placeholder: implemented in Phase" in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(REPO)))
    assert not hits, hits
