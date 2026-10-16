"""Static security checks run as tests (spec S.12 row 0, G.4).

* The real repository passes the forbidden-pattern scan, bandit and the
  worker-isolation check.
* Each scanner rule actually fires on a deliberately bad sample (otherwise a
  broken scanner would "pass" everything).

Bad samples are assembled from string pieces at runtime so that this test file
itself never contains a committed join code or private-key header.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.security

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_forbidden = _load("check_forbidden")
check_worker_isolation = _load("check_worker_isolation")


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _rules(root: Path) -> list[str]:
    findings, _ = check_forbidden.scan(root)
    return [f"{f.path}:{f.rule}" for f in findings]


# ----------------------------------------------------------- real repository


def test_repository_forbidden_scan_is_clean() -> None:
    findings, _exemptions = check_forbidden.scan(REPO)
    assert findings == []


def test_repository_worker_isolation_is_clean() -> None:
    assert check_worker_isolation.check(REPO) == []


def test_bandit_clean() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-q",
            "-r",
            "mdc",
            "scripts",
            "-c",
            "pyproject.toml",
            "-ll",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_check_forbidden_cli_exit_codes(tmp_path: Path) -> None:
    clean = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_forbidden.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert clean.returncode == 0
    _write(tmp_path, "mdc/bad.py", "import pickle\n")
    dirty = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "check_forbidden.py"),
            "--root",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert dirty.returncode == 1
    assert '"violations"' in dirty.stdout


# ------------------------------------------------------- python rules fire


@pytest.mark.parametrize(
    ("source", "rule_fragment"),
    [
        ("import pickle\n", "banned module 'pickle'"),
        ("import marshal\n", "banned module 'marshal'"),
        ("from shelve import open as o\n", "banned module 'shelve'"),
        ("x = eval('1')\n", "call to eval()"),
        ("exec('x=1')\n", "call to exec()"),
        ("m = __import__('os')\n", "call to __import__()"),
        ("import yaml\nyaml.load('a')\n", "yaml.load"),
        ("import subprocess\nsubprocess.run('ls', shell=True)\n", "shell=True"),
        ("import os\nos.system('ls')\n", "os.system"),
        ("import os\nos.popen('ls')\n", "os.popen"),
        ("import tempfile\ntempfile.mktemp()\n", "mktemp"),
        ("import os\nos.chmod('f', 0o777)\n", "world-writable"),
        ("import os\nos.chmod('f', 0o666)\n", "world-writable"),
        ("import ssl\nx = ssl.CERT_NONE\n", "CERT_NONE"),
        ("ctx = object()\nctx.check_hostname = False\n", "check_hostname"),
        ("def f(mac, other):\n    return mac == other\n", "secret-like name 'mac'"),
        ("def f(device_secret, x):\n    return x != device_secret\n", "'device_secret'"),
        ("def f(self, t):\n    return self.proof == t\n", "'proof'"),
        ("assert 1 + 1 == 2\n", "assert in non-test code"),
        (
            "import logging\nlog = logging.getLogger()\ndef f(token):\n    log.info('%s', token)\n",
            "logging call with secret-like variable 'token'",
        ),
        (
            "import logging\ndef f(join_code):\n    logging.warning('x %s', join_code)\n",
            "join_code",
        ),
    ],
)
def test_python_rules_fire(tmp_path: Path, source: str, rule_fragment: str) -> None:
    _write(tmp_path, "mdc/sample.py", source)
    rules = _rules(tmp_path)
    assert any(rule_fragment in r for r in rules), rules


def test_random_banned_only_in_security_modules(tmp_path: Path) -> None:
    _write(tmp_path, "mdc/common/auth.py", "import random\nrandom.random()\n")
    _write(tmp_path, "mdc/common/ids.py", "from random import choice\n")
    _write(tmp_path, "mdc/experiment_ok.py", "import random\nrng = random.Random(1)\n")
    rules = _rules(tmp_path)
    assert sum("mdc/common/auth.py" in r for r in rules) == 2  # import + call
    assert any("mdc/common/ids.py" in r for r in rules)
    assert not any("experiment_ok" in r for r in rules)


def test_tls_module_may_disable_ca_verification(tmp_path: Path) -> None:
    src = "import ssl\nctx = ssl.create_default_context()\nctx.check_hostname = False\n"
    src += "ctx.verify_mode = ssl.CERT_NONE\n"
    _write(tmp_path, "mdc/common/tls.py", src)
    assert _rules(tmp_path) == []


def test_todo_banned_in_security_critical_modules_only(tmp_path: Path) -> None:
    _write(tmp_path, "mdc/common/protocol.py", "# TODO: tighten this\n")
    _write(tmp_path, "mdc/coordinator/other.py", "# TODO: fine here\n")
    rules = _rules(tmp_path)
    assert rules == ["mdc/common/protocol.py:TODO/FIXME in security-critical module"]


def test_safe_constructs_not_flagged(tmp_path: Path) -> None:
    src = (
        "import asyncio, hmac, yaml, subprocess, logging\n"
        "async def f():\n"
        "    await asyncio.create_subprocess_exec('ls')\n"
        "def g(mac, other, tokens, n_tokens):\n"
        "    ok = hmac.compare_digest(mac, other)\n"
        "    left = tokens == 0\n"  # 'tokens' is not the secret word 'token'
        "    sized = len(mac) == MAC_HEX_LEN or SECRET_BYTES != 32\n"  # public constants
        "    subprocess.run(['ls'], shell=False, timeout=5)\n"
        "    logging.getLogger().info('tokens left %d', n_tokens)\n"
        "    return yaml.safe_load('a: 1'), ok, left, sized, mac is None\n"
    )
    _write(tmp_path, "mdc/safe.py", src)
    _write(tmp_path, "tests/test_x.py", "assert True\nimport pickle\n")  # tests not scanned
    assert _rules(tmp_path) == []


def test_exemption_comment_is_listed_not_failed(tmp_path: Path) -> None:
    _write(tmp_path, "mdc/x.py", "assert True  # forbidden-ok: demonstration only\n")
    findings, exemptions = check_forbidden.scan(tmp_path)
    assert findings == []
    assert len(exemptions) == 1
    assert exemptions[0].rule.startswith("assert")


def test_syntax_error_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "scripts/broken.py", "def f(:\n")
    assert any("syntax error" in r for r in _rules(tmp_path))


# --------------------------------------------------------- static web rules


@pytest.mark.parametrize(
    ("name", "content", "fragment"),
    [
        ("app.js", "el.innerHTML = x;\n", "innerHTML"),
        ("app.js", "el.outerHTML = x;\n", "outerHTML"),
        ("app.js", "el.insertAdjacentHTML('beforeend', x);\n", "insertAdjacentHTML"),
        ("app.js", "document.write(x);\n", "document.write"),
        ("app.js", "eval(x);\n", "eval("),
        ("app.js", "const f = new Function('return 1');\n", "new Function"),
        ("app.js", 'setTimeout("alert(1)", 10);\n', "setTimeout"),
        ("app.js", "fetch('https' + '://evil.example');\n", None),
        ("app.js", "fetch('http://cdn.example/x.js');\n", "external URL"),
        ("index.html", '<button onclick="go()">x</button>\n', "inline event handler"),
        ("index.html", "<script>alert(1)</script>\n", "inline <script>"),
        ("index.html", '<script src="https://cdn.example/x.js"></script>\n', "external URL"),
        ("style.css", "@import url(http://fonts.example/x.css);\n", "external URL"),
    ],
)
def test_static_rules_fire(tmp_path: Path, name: str, content: str, fragment: str | None) -> None:
    _write(tmp_path, f"mdc/dashboard/static/{name}", content)
    rules = _rules(tmp_path)
    if fragment is None:
        assert rules == []  # a URL split across string pieces is not a literal URL
    else:
        assert any(fragment in r for r in rules), rules


def test_static_safe_html_not_flagged(tmp_path: Path) -> None:
    html = '<!doctype html>\n<script src="app.js"></script>\n<p>online</p>\n'
    js = "el.textContent = name;\nsetTimeout(tick, 500);\n"
    _write(tmp_path, "mdc/dashboard/static/index.html", html)
    _write(tmp_path, "mdc/dashboard/static/app.js", js)
    assert _rules(tmp_path) == []


# ---------------------------------------------------------- repo secrets


def test_committed_join_code_detected(tmp_path: Path) -> None:
    code = "MDC1" + "." + "eyJ2IjoxLCJob3N0IjoiMTI3LjAuMC4xIn0abcdefgh"
    _write(tmp_path, "notes.md", f"my code: {code}\n")
    _write(tmp_path, "phone.join", code + "\n")
    rules = _rules(tmp_path)
    assert "notes.md:committed join code" in rules
    assert "phone.join:committed join code" in rules


def test_committed_private_key_detected_but_fixtures_allowed(tmp_path: Path) -> None:
    pem = "-----BEGIN " + "EC PRIVATE KEY-----\nMHcCAQEE\n-----END EC PRIVATE KEY-----\n"
    _write(tmp_path, "configs/key.pem", pem)
    _write(tmp_path, "tests/fixtures/generated.pem", pem)
    _write(tmp_path, ".venv/lib/x.pem", pem)  # skipped directory
    rules = _rules(tmp_path)
    assert rules == ["configs/key.pem:committed PEM private key"]


def test_short_mdc1_mentions_are_not_join_codes(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "Frames start with the magic MDC1. Join codes look like MDC1.x\n")
    assert _rules(tmp_path) == []


# ------------------------------------------------------- worker isolation


@pytest.mark.parametrize(
    ("rel", "source", "fragment"),
    [
        ("mdc/worker/w.py", "from mdc.coordinator import server\n", "mdc.coordinator"),
        ("mdc/worker/w.py", "import mdc.dashboard.web\n", "mdc.dashboard"),
        ("mdc/worker/w.py", "from ..coordinator import job\n", "mdc.coordinator"),
        ("mdc/tasks/t.py", "import PIL\n", "third-party package 'PIL'"),
        ("mdc/common/c.py", "import yaml\n", "third-party package 'yaml'"),
        ("mdc/common/__init__.py", "from ..dashboard import x\n", "mdc.dashboard"),
        ("mdc/worker/w.py", "import mdc.commonx\n", "mdc.commonx"),
    ],
)
def test_isolation_violations_detected(
    tmp_path: Path, rel: str, source: str, fragment: str
) -> None:
    _write(tmp_path, rel, source)
    problems = check_worker_isolation.check(tmp_path)
    assert any(fragment in p for p in problems), problems


def test_isolation_allows_stdlib_numpy_and_shared_code(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "mdc/worker/w.py",
        "from __future__ import annotations\nimport asyncio, ssl, json\nimport numpy as np\n"
        "from mdc.common import ewma\nfrom ..tasks import base\nfrom . import executor\n"
        "import mdc\n",
    )
    _write(tmp_path, "mdc/common/config.py", "import yaml\n")  # documented exception
    _write(tmp_path, "mdc/coordinator/c.py", "import aiohttp\n")  # not worker side
    assert check_worker_isolation.check(tmp_path) == []
