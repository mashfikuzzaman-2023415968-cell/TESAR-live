"""Worker isolation check (spec G.2 step 8, S.11).

Code that ships to phones (``mdc/worker``, ``mdc/common``, ``mdc/tasks``) may
import only:

* the Python standard library,
* ``numpy``,
* other modules of ``mdc.worker``, ``mdc.common``, ``mdc.tasks`` (and ``mdc``).

It must never import ``mdc.coordinator`` or ``mdc.dashboard`` (they are not in
the worker zip and pull in laptop-only libraries), nor any other third-party
package (Termux only has numpy installed).

Single documented exception (DECISIONS.md, 2026-10-07): ``mdc/common/config.py``
is the coordinator's YAML config loader and imports PyYAML. Workers never import
it; a test checks that importing the worker-side modules does not load ``yaml``.

Usage: ``python scripts/check_worker_isolation.py [--root DIR]``
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

WORKER_SIDE = ("mdc/worker", "mdc/common", "mdc/tasks")
ALLOWED_THIRD_PARTY = {"numpy"}
ALLOWED_FIRST_PARTY = ("mdc.worker", "mdc.common", "mdc.tasks")
# Documented exceptions: file -> extra allowed top-level packages.
EXCEPTIONS: dict[str, set[str]] = {"mdc/common/config.py": {"yaml"}}


def _resolve_relative(rel_path: str, module: str | None, level: int) -> str:
    """Turn ``from ..x import y`` inside ``rel_path`` into an absolute name."""
    package_parts = Path(rel_path).with_suffix("").parts[:-1]  # drop the module name
    if Path(rel_path).name == "__init__.py":
        package_parts = Path(rel_path).parent.parts
    base = (
        list(package_parts[: len(package_parts) - (level - 1)])
        if level > 1
        else list(package_parts)
    )
    return ".".join([*base, module] if module else base)


def _is_allowed_first_party(name: str) -> bool:
    """``mdc``, or ``mdc.common``/``mdc.tasks``/``mdc.worker`` and their submodules.

    Compared component-wise so that e.g. ``mdc.commonx`` does not pass.
    """
    return name == "mdc" or any(
        name == allowed or name.startswith(allowed + ".") for allowed in ALLOWED_FIRST_PARTY
    )


def _is_allowed_external(top: str, extra: set[str]) -> bool:
    """Standard library, ``__future__``, numpy, or a documented per-file exception."""
    return (
        top in sys.stdlib_module_names
        or top == "__future__"
        or top in ALLOWED_THIRD_PARTY
        or top in extra
    )


def check_file(path: Path, rel: str) -> list[str]:
    """Return human-readable problems for one file."""
    problems: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    extra = EXCEPTIONS.get(rel, set())
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names = [_resolve_relative(rel, node.module, node.level)]
            else:
                names = [node.module or ""]
        for name in names:
            top = name.split(".")[0]
            if top == "mdc":
                if not _is_allowed_first_party(name):
                    problems.append(
                        f"{rel}:{node.lineno}: imports {name} (coordinator/dashboard code)"
                    )
            elif not _is_allowed_external(top, extra):
                problems.append(f"{rel}:{node.lineno}: imports third-party package '{top}'")
    return problems


def check(root: Path) -> list[str]:
    """Check every worker-side file under ``root``."""
    problems: list[str] = []
    for top in WORKER_SIDE:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            problems.extend(check_file(path, path.relative_to(root).as_posix()))
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 1 if any forbidden import is found."""
    parser = argparse.ArgumentParser(description="Worker isolation check")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = parser.parse_args(argv)
    problems = check(Path(args.root).resolve())
    for problem in problems:
        print(f"ISOLATION  {problem}")
    print(f"check_worker_isolation: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
