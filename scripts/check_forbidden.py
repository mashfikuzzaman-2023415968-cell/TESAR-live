"""Forbidden-pattern scanner (spec G.4).

Scans the code base for constructs that are banned for security reasons and
exits non-zero if any are found. Three groups of rules:

1. **Python** (``mdc/``, ``scripts/``, ``experiments/``): most rules use the
   ``ast`` module rather than text search. That is more precise: for example a
   text search for the built-in ``exec`` call would also match
   ``asyncio.create_subprocess_exec``, while the AST only matches a call to the
   bare built-in name.
2. **Dashboard frontend** (``mdc/dashboard/static/``): regular expressions for
   DOM-injection sinks, inline handlers/scripts and external URLs.
3. **Repository-wide**: committed join codes or PEM private keys.

An individual line can be exempted with an inline comment
``# forbidden-ok: <reason>`` (``// forbidden-ok: <reason>`` in JS); every
exemption is listed in the output so reviewers see it.

Usage: ``python scripts/check_forbidden.py [--root DIR] [--json]``
Exit status: 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PY_DIRS = ("mdc", "scripts", "experiments")
STATIC_DIR = "mdc/dashboard/static"
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "runs",
    "dist",
    "build",
    "node_modules",
}
# Generated at test time only; never committed.
FIXTURE_DIR = "tests/fixtures"
MAX_SCAN_BYTES = 5_000_000  # larger files are data, not source

# The only module allowed to disable CA verification (it pins certificates instead).
TLS_MODULE = "mdc/common/tls.py"
# Modules where the non-cryptographic ``random`` module is banned outright.
NO_RANDOM_MODULES = {
    "mdc/common/auth.py",
    "mdc/common/tls.py",
    "mdc/common/credentials.py",
    "mdc/common/ids.py",
    "mdc/common/ratelimit.py",
    "mdc/coordinator/enrollment.py",
    "mdc/dashboard/auth.py",
}
# Security-critical modules: no TODO/FIXME markers allowed.
SECURITY_CRITICAL = NO_RANDOM_MODULES | {
    "mdc/common/protocol.py",
    "mdc/common/validation.py",
    "mdc/common/redact.py",
    "mdc/common/safe_names.py",
    "mdc/common/statedir.py",
    "mdc/coordinator/certs.py",
    "mdc/coordinator/verifier.py",
    "mdc/coordinator/security_log.py",
    "mdc/dashboard/headers.py",
}

BANNED_MODULES = {"pickle", "marshal", "shelve"}
BANNED_BUILTIN_CALLS = {"eval", "exec", "__import__"}
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
# Name *components* (split on "_") that mark a value as secret-like.
SECRET_PARTS = {"mac", "digest", "token", "secret", "proof", "fingerprint"}
LOG_SECRET_PARTS = {"secret", "token"}

EXEMPT_RE = re.compile(r"(#|//)\s*forbidden-ok:\s*\S")
TODO_RE = re.compile(r"\b(TODO|FIXME)\b")
# Built by concatenation so this file does not itself contain the patterns.
JOIN_CODE_RE = re.compile("MDC1" + r"\.[A-Za-z0-9_-]{24,}")
PEM_KEY_RE = re.compile("-----BEGIN " + r"[A-Z ]*" + "PRIVATE KEY-----")

JS_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("innerHTML", re.compile(r"\binnerHTML\b")),
    ("outerHTML", re.compile(r"\bouterHTML\b")),
    ("insertAdjacentHTML", re.compile(r"\binsertAdjacentHTML\b")),
    ("document.write", re.compile(r"\bdocument\s*\.\s*write(ln)?\b")),
    ("eval(", re.compile(r"\beval\s*\(")),
    ("new Function(", re.compile(r"\bnew\s+Function\s*\(")),
    ("setTimeout/setInterval with string", re.compile(r"\bset(Timeout|Interval)\s*\(\s*[\"'`]")),
    ("external URL", re.compile(r"https?://")),
]
HTML_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("inline event handler attribute", re.compile(r"<[^>]*\son[a-z]+\s*=", re.IGNORECASE)),
    ("external URL", re.compile(r"https?://")),
]
INLINE_SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".js",
    ".html",
    ".css",
    ".sh",
    ".cfg",
    ".ini",
    ".join",
    ".pem",
    ".key",
    "",
}


@dataclass
class Finding:
    """One violation (or one reviewed exemption)."""

    path: str
    line: int
    rule: str
    text: str


def _name_parts(name: str) -> set[str]:
    """``"device_secret"`` -> ``{"device", "secret"}`` (lower case)."""
    return {p for p in name.lower().split("_") if p}


def _expr_name(node: ast.AST) -> str | None:
    """Return the identifier of a Name or the attribute name of an Attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dotted(node: ast.AST) -> str:
    """``os.system`` -> ``"os.system"`` (best effort; '' if not a plain dotted name)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


class _PyVisitor(ast.NodeVisitor):
    """Collects (line, rule) pairs for the Python AST rules."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.hits: list[tuple[int, str]] = []
        self.in_mdc = rel.startswith("mdc/")
        self.no_random = rel in NO_RANDOM_MODULES
        self.is_tls = rel == TLS_MODULE

    def _hit(self, node: ast.AST, rule: str) -> None:
        self.hits.append((getattr(node, "lineno", 1), rule))

    # -- imports ----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BANNED_MODULES:
                self._hit(node, f"import of banned module '{top}'")
            if self.no_random and top == "random":
                self._hit(node, "'random' module in a security module (use secrets)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if node.level == 0 and top in BANNED_MODULES:
            self._hit(node, f"import of banned module '{top}'")
        if node.level == 0 and self.no_random and top == "random":
            self._hit(node, "'random' module in a security module (use secrets)")
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        dotted = _dotted(func)
        if isinstance(func, ast.Name) and func.id in BANNED_BUILTIN_CALLS:
            self._hit(node, f"call to {func.id}()")
        if dotted == "yaml.load" or dotted.endswith(".yaml.load"):
            self._hit(node, "yaml.load (use yaml.safe_load)")
        if dotted in {"os.system", "os.popen"}:
            self._hit(node, f"{dotted}()")
        if dotted in {"tempfile.mktemp", "mktemp"}:
            self._hit(node, "tempfile.mktemp (race-prone; use mkstemp)")
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self._hit(node, "subprocess with shell=True")
        if _expr_name(func) == "chmod":
            for arg in node.args[1:]:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, int)
                    and not isinstance(arg.value, bool)
                    and arg.value & 0o002
                ):
                    self._hit(node, f"world-writable chmod mode {arg.value:o}")
        if isinstance(func, ast.Attribute) and func.attr in LOG_METHODS:
            for sub in ast.walk(ast.Tuple(elts=list(node.args), ctx=ast.Load())):
                name = _expr_name(sub)
                if name and (_name_parts(name) & LOG_SECRET_PARTS or "join_code" in name.lower()):
                    self._hit(node, f"logging call with secret-like variable '{name}'")
                    break
        if self.no_random and dotted.startswith("random."):
            self._hit(node, "'random' module in a security module (use secrets)")
        self.generic_visit(node)

    # -- TLS verification ---------------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "CERT_NONE" and not self.is_tls:
            self._hit(node, f"ssl.CERT_NONE outside {TLS_MODULE}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                _expr_name(target) == "check_hostname"
                and isinstance(node.value, ast.Constant)
                and node.value.value is False
                and not self.is_tls
            ):
                self._hit(node, f"check_hostname = False outside {TLS_MODULE}")
        self.generic_visit(node)

    # -- comparisons and asserts -------------------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            for operand in [node.left, *node.comparators]:
                name = _expr_name(operand)
                # UPPER_CASE names are public constants (e.g. SECRET_BYTES = 32),
                # not secret values, so comparing against them is fine.
                if name and not name.isupper() and _name_parts(name) & SECRET_PARTS:
                    self._hit(
                        node,
                        f"==/!= on secret-like name '{name}' (use hmac.compare_digest)",
                    )
                    break
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self.in_mdc:
            self._hit(node, "assert in non-test code (stripped by python -O)")
        self.generic_visit(node)


def _iter_files(base: Path, suffixes: set[str]) -> list[Path]:
    """All files under ``base`` with one of ``suffixes``, skipping junk dirs."""
    if not base.exists():
        return []
    files: list[Path] = []
    for path in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in suffixes:
            files.append(path)
    return files


def _record(
    rel: str,
    lines: list[str],
    lineno: int,
    rule: str,
    findings: list[Finding],
    exemptions: list[Finding],
) -> None:
    text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
    item = Finding(rel, lineno, rule, text[:160])
    if EXEMPT_RE.search(text):
        exemptions.append(item)
    else:
        findings.append(item)


def scan_python(root: Path, findings: list[Finding], exemptions: list[Finding]) -> None:
    """Apply the Python rules to every ``.py`` file in PY_DIRS."""
    for top in PY_DIRS:
        for path in _iter_files(root / top, {".py"}):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            try:
                tree = ast.parse(source, filename=rel)
            except SyntaxError as exc:
                findings.append(Finding(rel, exc.lineno or 1, "syntax error", str(exc)))
                continue
            visitor = _PyVisitor(rel)
            visitor.visit(tree)
            for lineno, rule in visitor.hits:
                _record(rel, lines, lineno, rule, findings, exemptions)
            if rel in SECURITY_CRITICAL:
                for i, line in enumerate(lines, start=1):
                    if TODO_RE.search(line):
                        _record(
                            rel,
                            lines,
                            i,
                            "TODO/FIXME in security-critical module",
                            findings,
                            exemptions,
                        )


def scan_static(root: Path, findings: list[Finding], exemptions: list[Finding]) -> None:
    """Apply the JS/HTML/CSS rules to the dashboard's static files."""
    for path in _iter_files(root / STATIC_DIR, {".js", ".html", ".css"}):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        rules = JS_RULES if path.suffix == ".js" else HTML_RULES
        if path.suffix == ".css":
            rules = [("external URL", re.compile(r"https?://"))]
        for i, line in enumerate(lines, start=1):
            for rule, pattern in rules:
                if pattern.search(line):
                    _record(rel, lines, i, rule, findings, exemptions)
        if path.suffix == ".html":
            for match in INLINE_SCRIPT_RE.finditer(text):
                if match.group(2).strip():
                    lineno = text.count("\n", 0, match.start()) + 1
                    _record(rel, lines, lineno, "inline <script> code", findings, exemptions)


def scan_repo_secrets(root: Path, findings: list[Finding], exemptions: list[Finding]) -> None:
    """Find join codes and PEM private keys anywhere in the repository."""
    for path in _iter_files(root, TEXT_SUFFIXES):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(FIXTURE_DIR + "/"):
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if JOIN_CODE_RE.search(line):
                _record(rel, lines, i, "committed join code", findings, exemptions)
            if PEM_KEY_RE.search(line):
                _record(rel, lines, i, "committed PEM private key", findings, exemptions)


def scan(root: Path) -> tuple[list[Finding], list[Finding]]:
    """Run every rule under ``root``; return (violations, exemptions)."""
    findings: list[Finding] = []
    exemptions: list[Finding] = []
    scan_python(root, findings, exemptions)
    scan_static(root, findings, exemptions)
    scan_repo_secrets(root, findings, exemptions)
    return findings, exemptions


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Forbidden-pattern scanner (spec G.4)")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    findings, exemptions = scan(root)
    if args.json:
        print(
            json.dumps(
                {
                    "violations": [asdict(f) for f in findings],
                    "exemptions": [asdict(f) for f in exemptions],
                },
                indent=2,
            )
        )
    else:
        for f in exemptions:
            print(f"EXEMPT  {f.path}:{f.line}: {f.rule}: {f.text}")
        for f in findings:
            print(f"FORBID  {f.path}:{f.line}: {f.rule}: {f.text}")
        print(
            f"check_forbidden: {len(findings)} violation(s), {len(exemptions)} reviewed "
            "exemption(s)"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
