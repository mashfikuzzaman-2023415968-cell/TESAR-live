"""THE phase gate (spec Section G.2). Run after every phase.

Usage::

    python scripts/phase_gate.py --phase N [--repeat-e2e K] [--offline] [--keep-going] [--soak]

Runs, in order: environment checks, compile, format, lint, types, SAST,
forbidden patterns, worker isolation, dependency audit, unit/property/
integration tests with coverage, the security suite, e2e tests (phase >= 4),
perf smoke (phase >= 9) and soak (phase 14), per-module coverage of
security-critical modules, the skip/regression guard and phase-specific
checks. It stops at the first failure unless ``--keep-going`` is given, always
prints a summary table, writes ``runs/gates/phase-N.json``, appends a summary
to ``PROGRESS.md`` and finally prints ``PHASE N GATE: PASS`` or
``PHASE N GATE: FAIL (step X)``.

Every external tool is invoked as ``sys.executable -m <tool>`` with an
argument list (never a shell) and a timeout, so the gate uses exactly the
tools installed in the current virtual environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GATES_DIR = ROOT / "runs" / "gates"
BASELINE = GATES_DIR / "baseline.json"

# Per-module line coverage >= 95% for these, from the phase each is introduced.
SECURITY_CRITICAL_COVERAGE: dict[str, int] = {
    "mdc/common/protocol.py": 1,
    "mdc/common/auth.py": 1,
    "mdc/common/tls.py": 1,
    "mdc/common/validation.py": 1,
    "mdc/common/credentials.py": 1,
    "mdc/coordinator/enrollment.py": 1,
    "mdc/common/ratelimit.py": 4,
    "mdc/coordinator/verifier.py": 10,
    "mdc/dashboard/auth.py": 11,
}
CRITICAL_THRESHOLD = 95.0
MILESTONE_PHASES = {5, 10, 14}
TOOL_TIMEOUT_S = 1800


# Coverage thresholds T(N) from spec G.2: (last phase of band, percent).
COVERAGE_BANDS = ((3, 70), (7, 80), (14, 85))
E2E_FROM_PHASE = 4
PERF_FROM_PHASE = 9
SOAK_PHASE = 14
LAST_PHASE = 14
PYTEST_NO_TESTS = 5  # pytest exit code: no tests collected


def coverage_threshold(phase: int) -> int:
    """T(N) from spec G.2: 70% for phases 0-3, 80% for 4-7, 85% for 8-14."""
    for last_phase, percent in COVERAGE_BANDS:
        if phase <= last_phase:
            return percent
    return COVERAGE_BANDS[-1][1]


@dataclass
class StepResult:
    """Outcome of one gate step."""

    name: str
    status: str  # PASS | FAIL | SKIP
    duration_s: float
    notes: str = ""


@dataclass
class Gate:
    """Gate state shared by all steps."""

    phase: int
    repeat_e2e: int
    offline: bool
    keep_going: bool
    soak: bool
    results: list[StepResult] = field(default_factory=list)
    exemptions: list[str] = field(default_factory=list)
    result_files: list[Path] = field(default_factory=list)
    coverage_json: Path = GATES_DIR / "coverage.json"


class StepFailed(Exception):
    """Raised inside a step to mark it FAIL with a note."""


class StepSkipped(Exception):
    """Raised inside a step to mark it SKIP with a note."""


def run(
    cmd: list[str],
    timeout: float = TOOL_TIMEOUT_S,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a command (argument list, no shell), echoing its output."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(  # fixed argv of our own tools, no shell
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        print("    " + "\n    ".join(output.rstrip().splitlines()[-60:]), flush=True)
    return proc


def py(*args: str) -> list[str]:
    """``[sys.executable, *args]``."""
    return [sys.executable, *args]


def expect_ok(proc: subprocess.CompletedProcess[str], what: str) -> None:
    """Fail the step if the process exited non-zero."""
    if proc.returncode != 0:
        raise StepFailed(f"{what} exited with {proc.returncode}")


# ----------------------------------------------------------------------- steps


def step_environment(gate: Gate) -> str:
    """Python version, tools present, no stray secret files in the tree."""
    missing = []
    for module in ("ruff", "mypy", "bandit", "pytest"):
        proc = run(py("-m", module, "--version"), timeout=120)
        if proc.returncode != 0:
            missing.append(module)
    if missing:
        raise StepFailed(f"missing tools: {missing} (pip install -r requirements-dev.txt)")
    stray: list[str] = []
    skip = {".git", ".venv", "venv", "runs", "dist", "__pycache__", "node_modules"}
    for path in ROOT.rglob("*"):
        if any(part in skip for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        secret_like = path.suffix in {".join", ".pem", ".key"}
        if path.is_file() and secret_like and not rel.startswith("tests/fixtures/"):
            stray.append(rel)
    if stray:
        raise StepFailed(f"secret-like files in the working tree: {stray}")
    git = shutil.which("git")
    git_note = "git not found"
    if git and (ROOT / ".git").exists():
        proc = run([git, "status", "--porcelain"], timeout=60)
        git_note = f"git: {len(proc.stdout.splitlines())} changed/untracked path(s)"
    return f"Python {sys.version.split()[0]}; tools ok; {git_note}"


def step_compile(gate: Gate) -> str:
    """Byte-compile every Python file (catches syntax errors everywhere)."""
    targets = [d for d in ("mdc", "scripts", "experiments") if (ROOT / d).exists()]
    expect_ok(run(py("-m", "compileall", "-q", *targets)), "compileall")
    return ", ".join(targets)


def step_format(gate: Gate) -> str:
    """``ruff format --check .``"""
    expect_ok(run(py("-m", "ruff", "format", "--check", ".")), "ruff format")
    return "formatted"


def step_lint(gate: Gate) -> str:
    """``ruff check .`` with the rule sets configured in pyproject.toml."""
    expect_ok(run(py("-m", "ruff", "check", ".")), "ruff check")
    return "E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean"


def step_types(gate: Gate) -> str:
    """``mypy mdc`` (strict for security-critical packages)."""
    expect_ok(run(py("-m", "mypy", "mdc")), "mypy")
    return "mypy clean"


def step_sast(gate: Gate) -> str:
    """``bandit -r mdc scripts -c pyproject.toml -ll`` (medium/high fail)."""
    expect_ok(
        run(py("-m", "bandit", "-q", "-r", "mdc", "scripts", "-c", "pyproject.toml", "-ll")),
        "bandit",
    )
    return "no medium/high findings"


def step_forbidden(gate: Gate) -> str:
    """Forbidden-pattern scanner (G.4); exemptions are listed for review."""
    proc = run(py("scripts/check_forbidden.py", "--json"))
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise StepFailed("check_forbidden produced no JSON") from exc
    gate.exemptions = [f"{e['path']}:{e['line']} {e['rule']}" for e in data.get("exemptions", [])]
    if data.get("violations"):
        raise StepFailed(f"{len(data['violations'])} violation(s)")
    return f"clean; {len(gate.exemptions)} reviewed exemption(s)"


def step_isolation(gate: Gate) -> str:
    """Worker-side code imports nothing from coordinator/dashboard/third party."""
    expect_ok(run(py("scripts/check_worker_isolation.py")), "check_worker_isolation")
    return "worker side = stdlib + numpy"


def step_audit(gate: Gate) -> str:
    """``pip-audit`` against the lock files (needs network)."""
    if gate.offline:
        raise StepSkipped("offline: dependency audit skipped — run later without --offline")
    lock_files = ("requirements.lock", "requirements-worker.lock")
    locks = [p for p in lock_files if (ROOT / p).exists()]
    if len(locks) < len(lock_files):
        raise StepFailed("requirements.lock and requirements-worker.lock must both exist")
    args = ["-m", "pip_audit", "--progress-spinner", "off", "--disable-pip", "--require-hashes"]
    for lock in locks:
        args += ["-r", lock]
    expect_ok(run(py(*args), timeout=900), "pip-audit")
    return "no known vulnerabilities"


def _pytest(
    gate: Gate, label: str, marker: str, extra: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run pytest for one marker expression, recording outcomes as JSON.

    ``tests/conftest.py`` writes every test id and skip reason to the file named
    by ``MDC_GATE_RESULTS``; the skip guard reads those files.
    """
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    results = GATES_DIR / f"results-{label}.json"
    results.unlink(missing_ok=True)
    gate.result_files.append(results)
    env = dict(os.environ, MDC_GATE_RESULTS=str(results))
    cmd = py("-m", "pytest", "-q", "-m", marker, *extra)
    return run(cmd, timeout=7200, env=env)


def step_unit(gate: Gate) -> str:
    """Unit + property + integration tests with branch coverage >= T(N)."""
    threshold = coverage_threshold(gate.phase)
    proc = _pytest(
        gate,
        "unit",
        "not e2e and not soak and not perf",
        [
            "--cov=mdc",
            "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            f"--cov-report=json:{gate.coverage_json}",
            f"--cov-fail-under={threshold}",
        ],
    )
    expect_ok(proc, "pytest (unit/property/integration)")
    total = json.loads(gate.coverage_json.read_text())["totals"]["percent_covered"]
    return f"coverage {total:.1f}% (threshold {threshold}%)"


def step_security(gate: Gate) -> str:
    """The security suite (``pytest -m security``)."""
    proc = _pytest(gate, "security", "security", [])
    if proc.returncode == PYTEST_NO_TESTS:
        raise StepFailed("no security tests collected")
    expect_ok(proc, "pytest -m security")
    return "security suite green"


def step_e2e(gate: Gate) -> str:
    """End-to-end tests, repeated K times (K forced to 3 at milestones)."""
    if gate.phase < E2E_FROM_PHASE:
        raise StepSkipped("e2e tests start at Phase 4")
    repeat = max(gate.repeat_e2e, 3 if gate.phase in MILESTONE_PHASES else 1)
    for i in range(repeat):
        proc = _pytest(gate, f"e2e-{i + 1}", "e2e", [])
        if proc.returncode == PYTEST_NO_TESTS:
            raise StepFailed("no e2e tests collected")
        expect_ok(proc, f"pytest -m e2e (run {i + 1}/{repeat})")
    return f"e2e green x{repeat}"


def step_perf_soak(gate: Gate) -> str:
    """Perf smoke (phase >= 9) and soak (phase 14 or --soak)."""
    notes = []
    if gate.phase >= PERF_FROM_PHASE:
        proc = _pytest(gate, "perf", "perf", [])
        expect_ok(proc, "pytest -m perf")
        notes.append("perf ok")
    if gate.phase >= SOAK_PHASE or gate.soak:
        # The soak test sets @pytest.mark.timeout(0) itself; disabling the plugin with
        # -p no:timeout would unregister that marker and break --strict-markers collection.
        proc = _pytest(gate, "soak", "soak", [])
        expect_ok(proc, "pytest -m soak")
        notes.append("soak ok")
    if not notes:
        raise StepSkipped("perf from Phase 9, soak at Phase 14")
    return ", ".join(notes)


def step_critical_coverage(gate: Gate) -> str:
    """Per-module line coverage >= 95% for security-critical modules."""
    due = {m: p for m, p in SECURITY_CRITICAL_COVERAGE.items() if p <= gate.phase}
    if not due:
        raise StepSkipped("no security-critical module due yet")
    if not gate.coverage_json.exists():
        raise StepFailed("coverage.json missing (unit step did not run)")
    files = json.loads(gate.coverage_json.read_text())["files"]
    low = []
    notes = []
    for module in sorted(due):
        summary = files.get(module, {}).get("summary")
        if summary is None:
            low.append(f"{module}: not measured")
            continue
        statements = summary["num_statements"]
        covered = summary["covered_lines"]
        pct = 100.0 if statements == 0 else 100.0 * covered / statements
        notes.append(f"{Path(module).name} {pct:.0f}%")
        if pct < CRITICAL_THRESHOLD:
            low.append(f"{module}: {pct:.1f}%")
    if low:
        raise StepFailed("below 95%: " + "; ".join(low))
    return ", ".join(notes)


def _collect_results(files: list[Path]) -> tuple[set[str], dict[str, str]]:
    """Return (all test ids, skipped test id -> reason) from the gate's result files."""
    tests: set[str] = set()
    skipped: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for test_id, entry in data.items():
            tests.add(test_id)
            if entry.get("outcome") == "skipped":
                skipped[test_id] = entry.get("reason", "")
    return tests, skipped


def step_skip_guard(gate: Gate) -> str:
    """New skips or vanished tests fail the gate (tests are never deleted)."""
    tests, skipped = _collect_results(gate.result_files)
    baseline: dict[str, object] = {}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
    base_tests = set(baseline.get("tests", []))  # type: ignore[arg-type]
    base_skips = set(baseline.get("skipped", []))  # type: ignore[arg-type]
    decisions = (
        (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        if (ROOT / "DECISIONS.md").exists()
        else ""
    )
    problems = []
    vanished = sorted(base_tests - tests)
    if vanished:
        problems.append(f"{len(vanished)} test(s) from earlier gates are missing: {vanished[:5]}")
    for test_id, reason in sorted(skipped.items()):
        if test_id in base_skips:
            continue
        optional_ok = "pulp" in reason.lower() and "pulp" in decisions.lower()
        if not optional_ok:
            problems.append(f"new skip {test_id}: {reason}")
    if problems:
        raise StepFailed("; ".join(problems))
    # Baseline only ever grows in tests; skips may shrink.
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(
            {"phase": gate.phase, "tests": sorted(tests | base_tests), "skipped": sorted(skipped)},
            indent=1,
        )
    )
    return f"{len(tests)} tests, {len(skipped)} skipped (all known)"


# ------------------------------------------------------- phase-specific checks


def check_p0(gate: Gate) -> str:
    """P0: config loads; test images exist; state dir created with 0700/0600."""
    sys.path.insert(0, str(ROOT))
    from mdc.common.config import load_config  # noqa: PLC0415 - imported after sys.path fix
    from mdc.common.statedir import (  # noqa: PLC0415
        ensure_private_dir,
        posix_permissions_supported,
        write_private_file,
    )

    load_config(ROOT / "configs" / "default.yaml")
    images = ["test_512.png", "test_1024.png", "test_2048.png", "test_1000x1300.png"]
    missing = [i for i in images if not (ROOT / "data" / "images" / i).exists()]
    if missing:
        raise StepFailed(f"missing test images {missing}: run scripts/make_test_images.py")
    with tempfile.TemporaryDirectory() as tmp:
        state = ensure_private_dir(Path(tmp) / "state")
        secret = write_private_file(state / "probe", b"x")
        if posix_permissions_supported():
            dmode = state.stat().st_mode & 0o777
            fmode = secret.stat().st_mode & 0o777
            if dmode != 0o700 or fmode != 0o600:  # noqa: PLR2004
                raise StepFailed(f"state dir mode {dmode:o} / file mode {fmode:o}")
    return "config ok; 4 images; state dir 0700 / files 0600"


def check_p1(gate: Gate) -> str:
    """P1: temp cert; enroll a device; join-code round trip; TLS pin ok / wrong pin fails."""
    import asyncio  # noqa: PLC0415

    sys.path.insert(0, str(ROOT))
    from mdc.common import tls  # noqa: PLC0415
    from mdc.common.credentials import (  # noqa: PLC0415
        decode_join_code,
        encode_join_code,
        read_join_file,
        write_join_file,
    )
    from mdc.coordinator.certs import generate_certificate  # noqa: PLC0415
    from mdc.coordinator.enrollment import DeviceRegistry  # noqa: PLC0415

    async def handshake(cert_path: Path, key_path: Path, pin: str) -> bool:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.close()

        server = await asyncio.start_server(
            handle, "127.0.0.1", 0, ssl=tls.server_context(cert_path, key_path)
        )
        port = server.sockets[0].getsockname()[1]
        try:
            _, writer = await tls.open_pinned_connection("127.0.0.1", port, pin, 5)
        except tls.PinMismatchError:
            return False
        finally:
            server.close()
            await server.wait_closed()
        writer.close()
        return True

    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state"
        cert = generate_certificate(state, 30)
        code = DeviceRegistry(state).enroll("gate-check", "127.0.0.1", 50010, cert.fingerprint)
        if decode_join_code(encode_join_code(code)) != code:
            raise StepFailed("join code round trip changed the code")
        if read_join_file(write_join_file(Path(tmp) / "gate.join", code)) != code:
            raise StepFailed("join file round trip changed the code")
        if not asyncio.run(handshake(cert.cert_path, cert.key_path, cert.fingerprint)):
            raise StepFailed("TLS handshake with the correct pin failed")
        if asyncio.run(handshake(cert.cert_path, cert.key_path, "sha256:" + "0" * 64)):
            raise StepFailed("TLS handshake with a WRONG pin succeeded")
    return "cert ok; enroll ok; join code round trip; pin ok / wrong pin refused"


def check_p2(gate: Gate) -> str:
    """P2: ``cli reference`` on test_1000x1300.png + tiled assembly equality."""
    sys.path.insert(0, str(ROOT))
    import numpy as np  # noqa: PLC0415

    from mdc.coordinator.job import execute_inprocess  # noqa: PLC0415
    from mdc.coordinator.job_builder import (  # noqa: PLC0415
        build_image_job_from_array,
        load_image,
    )

    image_path = ROOT / "data" / "images" / "test_1000x1300.png"
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.png"
        proc = run(
            py(
                "-m",
                "mdc.cli",
                "reference",
                "--image",
                str(image_path),
                "--r",
                "2",
                "--iters",
                "8",
                "--out",
                str(ref_path),
            ),
            timeout=600,
        )
        expect_ok(proc, "mdc.cli reference")
        expected = load_image(ref_path)
        job = build_image_job_from_array(load_image(image_path), 128, 2, 8, 64)
        tiled = execute_inprocess(job, [1, 3, 8, 2])["output"]
    if not np.array_equal(tiled, expected):
        differing = int(np.count_nonzero(tiled != expected))
        raise StepFailed(f"tiled output differs from reference in {differing} pixels")
    return f"reference {expected.shape[1]}x{expected.shape[0]} == tiled ({len(job.tasks)} tasks)"


def check_p3(gate: Gate) -> str:
    """P3: worker ``--diag`` runs and prints no secret (output scanned for the join secret)."""
    import base64  # noqa: PLC0415

    sys.path.insert(0, str(ROOT))
    from mdc.common.credentials import encode_join_code, write_join_file  # noqa: PLC0415
    from mdc.coordinator.certs import generate_certificate  # noqa: PLC0415
    from mdc.coordinator.enrollment import DeviceRegistry  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state"
        cert = generate_certificate(state, 30)
        code = DeviceRegistry(state).enroll("gate-diag", "127.0.0.1", 50010, cert.fingerprint)
        join = write_join_file(Path(tmp) / "gate.join", code)
        proc = run(
            py(
                "-m",
                "mdc.worker",
                "--diag",
                "--join-file",
                str(join),
                "--battery",
                "mock",
                "--diag-listen-s",
                "0.5",
                "--discovery-port",
                "0",
            ),
            timeout=300,
            env={**os.environ, "HOME": tmp},
        )
        expect_ok(proc, "mdc.worker --diag")
        output = proc.stdout + proc.stderr
        secret_forms = [
            code.secret.hex(),
            base64.urlsafe_b64encode(code.secret).decode().rstrip("="),
            base64.b64encode(code.secret).decode(),
            encode_join_code(code),
            "MDC1.",
        ]
        leaked = [f for f in secret_forms if f in output]
        if leaked:
            raise StepFailed("worker --diag output contains secret material")
        if "benchmark" not in output:
            raise StepFailed("worker --diag output incomplete")
    return "worker --diag ok, no secret in output"


def _free_port() -> int:
    import socket  # noqa: PLC0415

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_headless_job(
    tmp: Path,
    n_workers: int,
    overrides: list[str],
    worker_flags: dict[int, list[str]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Run ``python -m mdc.coordinator --headless --run-job --exit-after-job`` with
    ``n_workers`` enrolled worker processes on this machine; return report.json
    (and append the run's events to ``events`` if given)."""
    sys.path.insert(0, str(ROOT))
    from mdc.common.credentials import write_join_file  # noqa: PLC0415
    from mdc.coordinator.certs import ensure_certificate  # noqa: PLC0415
    from mdc.coordinator.enrollment import DeviceRegistry  # noqa: PLC0415

    state, out = tmp / "state", tmp / "out"
    port = _free_port()
    cert = ensure_certificate(state, 30)
    registry = DeviceRegistry(state)
    env = {**os.environ, "HOME": str(tmp), "PYTHONPATH": str(ROOT)}
    sets = [
        f"security.state_dir={state}",
        f"network.tcp_port={port}",
        "network.coordinator_bind=127.0.0.1",
        "network.beacon_modes=[]",
        *overrides,
    ]
    coord_cmd = py(
        "-m",
        "mdc.coordinator",
        "--headless",
        "--run-job",
        "--exit-after-job",
        "--wait-workers",
        str(n_workers),
        "--wait-timeout-s",
        "60",
        "--output-dir",
        str(out),
        *[a for kv in sets for a in ("--set", kv)],
    )
    workers: list[subprocess.Popen[bytes]] = []
    coord = subprocess.Popen(
        coord_cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        for i in range(n_workers):
            code = registry.enroll(f"gate-w{i}", "127.0.0.1", port, cert.fingerprint)
            join = write_join_file(tmp / f"w{i}.join", code)
            flags = (worker_flags or {}).get(i, [])
            workers.append(
                subprocess.Popen(
                    py(
                        "-m",
                        "mdc.worker",
                        "--join-file",
                        str(join),
                        "--coordinator",
                        f"127.0.0.1:{port}",
                        "--battery",
                        "mock",
                        "--log-dir",
                        str(tmp / f"log{i}"),
                        *flags,
                    ),
                    cwd=tmp,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        output, _ = coord.communicate(timeout=300)
    finally:
        for proc in [coord, *workers]:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    report_path = out / "report.json"
    if not report_path.exists():
        tail = output.decode(errors="replace")[-800:]
        raise StepFailed(f"no report.json (coordinator exit {coord.returncode}): {tail}")
    report: dict[str, object] = json.loads(report_path.read_text())
    if events is not None:
        lines = (out / "events.jsonl").read_text().splitlines()
        events.extend(json.loads(line) for line in lines if line.strip())
    return report


def check_p4(gate: Gate) -> str:
    """P4: headless 512 x 512 job with 2 local workers -> output_correct: true."""
    with tempfile.TemporaryDirectory() as tmp:
        report = run_headless_job(
            Path(tmp),
            2,
            [
                f"job.image_path={ROOT / 'data/images/test_512.png'}",
                "job.blur_radius=1",
                "job.blur_iterations=2",
                "scheduler.policy=round_robin",
            ],
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    return (
        f"512x512 job, {report['n_workers_remote']} workers + local: output_correct "
        f"(makespan {report['makespan_s']} s)"
    )


FAST_LIVENESS = [
    "liveness.ping_interval_s=0.25",
    "liveness.suspect_after_s=1.0",
    "liveness.dead_after_s=2.0",
    "liveness.task_timeout_min_s=1.0",
]


def check_p5(gate: Gate) -> str:
    """P5: the P4 job with a crashing and a dropping worker -> output correct."""
    with tempfile.TemporaryDirectory() as tmp:
        report = run_headless_job(
            Path(tmp),
            2,
            [
                f"job.image_path={ROOT / 'data/images/test_512.png'}",
                "job.blur_radius=1",
                "job.blur_iterations=2",
                "scheduler.policy=round_robin",
                *FAST_LIVENESS,
            ],
            worker_flags={
                0: ["--lab-mode", "--crash-after-s", "2", "--net-delay-ms", "300"],
                1: ["--lab-mode", "--drop-prob", "1.0"],
            },
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    return (
        f"crash + drop workers: output_correct, {report['attempts_timeout']} timeouts, "
        f"{report['attempts_lost']} lost, makespan {report['makespan_s']} s"
    )


def run_cluster_job(
    tmp: Path, profile: str, n_workers: int, overrides: list[str]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Coordinator CLI + ``scripts/local_cluster.py PROFILE``; return (report, events)."""
    state, out = tmp / "state", tmp / "out"
    port = _free_port()
    env = {**os.environ, "HOME": str(tmp), "PYTHONPATH": str(ROOT)}
    sets = [
        f"security.state_dir={state}",
        f"network.tcp_port={port}",
        "network.coordinator_bind=127.0.0.1",
        "network.beacon_modes=[]",
        "network.max_connections_per_ip=16",
        *overrides,
    ]
    coord = subprocess.Popen(
        py(
            "-m",
            "mdc.coordinator",
            "--headless",
            "--run-job",
            "--exit-after-job",
            "--wait-workers",
            str(n_workers),
            "--wait-timeout-s",
            "60",
            "--output-dir",
            str(out),
            *[a for kv in sets for a in ("--set", kv)],
        ),
        cwd=tmp,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    cluster = None
    try:
        time.sleep(1.0)  # let the coordinator create its certificate first
        cluster = subprocess.Popen(
            py(
                str(ROOT / "scripts" / "local_cluster.py"),
                str(ROOT / profile),
                "--coordinator",
                f"127.0.0.1:{port}",
                "--state-dir",
                str(state),
                "--runs-dir",
                str(tmp / "runs"),
                "--duration",
                "240",
            ),
            cwd=tmp,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output, _ = coord.communicate(timeout=300)
    finally:
        for proc in (coord, cluster):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    if not (out / "report.json").exists():
        tail = output.decode(errors="replace")[-800:]
        raise StepFailed(f"no report.json (coordinator exit {coord.returncode}): {tail}")
    report: dict[str, object] = json.loads((out / "report.json").read_text())
    events = [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines() if line.strip()
    ]
    return report, events


def check_p6(gate: Gate) -> str:
    """P6: hetero5 job with the ``fastest`` policy -> correct; attempt_sent has the new fields."""
    n_workers = 5  # hetero5
    with tempfile.TemporaryDirectory() as tmp:
        report, events = run_cluster_job(
            Path(tmp),
            "configs/cluster_profiles/hetero5.yaml",
            n_workers,
            [
                f"job.image_path={ROOT / 'data/images/test_512.png'}",
                "job.blur_iterations=4",
                "scheduler.policy=fastest",
                "chunking.target_task_s=0.1",
            ],
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    sent = [e for e in events if e["type"] == "attempt_sent"]
    needed = {"n_tiles", "predicted_s", "est_rtt_s", "est_bw_bytes_per_s"}
    if not sent or any(not needed <= set(e) for e in sent):
        raise StepFailed("attempt_sent events lack the Phase-6 fields")
    probes = [e for e in events if e["type"] == "probe_done"]
    if len(probes) < n_workers:
        raise StepFailed(f"only {len(probes)} probe_done events")
    sizes = sorted({int(e["n_tiles"]) for e in sent})  # type: ignore[call-overload]
    return (
        f"hetero5/fastest: output_correct, {len(sent)} attempts, chunk sizes {sizes}, "
        f"{len(probes)} probes"
    )


def _per_worker(report: dict[str, object]) -> dict[str, dict[str, Any]]:
    per_worker = report.get("per_worker")
    if not isinstance(per_worker, dict):
        raise StepFailed("report.json has no per_worker section")
    return per_worker


def check_p7(gate: Gate) -> str:
    """P7: hetero_faulty6 job -> correct; attempt_sent has predicted_R; unreliable
    workers end with a lower reputation than the stable ones."""
    with tempfile.TemporaryDirectory() as tmp:
        report, events = run_cluster_job(
            Path(tmp),
            "configs/cluster_profiles/hetero_faulty6.yaml",
            6,
            [
                f"job.image_path={ROOT / 'data/images/test_1024.png'}",
                "job.blur_iterations=6",
                "scheduler.policy=round_robin",  # every worker gets a long history
                "chunking.fixed_chunk_tiles=1",
                *FAST_LIVENESS,
            ],
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    sent = [e for e in events if e["type"] == "attempt_sent"]
    if not sent or any(not isinstance(e.get("predicted_R"), float) for e in sent):
        raise StepFailed("attempt_sent events lack predicted_R")
    reps = {name: w["rep_final"] for name, w in _per_worker(report).items()}
    if any(not isinstance(v, float) or not 0 < v < 1 for v in reps.values()):
        raise StepFailed(f"rep_final missing or out of range: {reps}")
    stable = min(reps[n] for n in ("fast-phone", "slow-phone", "laptop-b"))
    unreliable = min(reps[n] for n in ("flaky-phone", "dropper-phone", "crasher-phone"))
    if unreliable >= stable:
        raise StepFailed(f"unreliable workers do not rank lower: {reps}")
    shown = ", ".join(f"{n} {v:.2f}" for n, v in sorted(reps.items()))
    return f"hetero_faulty6/round_robin: output_correct, rep_final {shown}"


def check_p8(gate: Gate) -> str:
    """P8: a worker at 15 % battery (not charging) gets no task and is logged; the
    job is correct; attempts carry predicted energy and the report has energy fields."""
    events: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        report = run_headless_job(
            Path(tmp),
            3,
            [
                f"job.image_path={ROOT / 'data/images/test_512.png'}",
                "job.blur_iterations=2",
                "scheduler.policy=round_robin",
            ],
            worker_flags={
                0: ["--device-class", "phone", "--mock-battery-pct", "15"],
                1: ["--device-class", "phone", "--mock-battery-pct", "90"],
            },
            events=events,
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    low = _per_worker(report)["gate-w0"]
    if low["tasks_ok"] or any(
        e.get("worker_id") == low["worker_id"] for e in events if e["type"] == "attempt_sent"
    ):
        raise StepFailed("the 15 % worker was given work")
    logged = [e for e in events if e["type"] == "worker_ineligible_battery"]
    if [e["worker_id"] for e in logged] != [low["worker_id"]]:
        raise StepFailed(f"worker_ineligible_battery events: {logged}")
    sent = [e for e in events if e["type"] == "attempt_sent"]
    if not sent or any(not isinstance(e.get("predicted_energy_j"), float) for e in sent):
        raise StepFailed("attempt_sent events lack predicted_energy_j")
    phone = _per_worker(report)["gate-w1"]
    if not phone.get("energy_pred_j") or not phone.get("energy_budget_j"):
        raise StepFailed(f"no energy figures for gate-w1: {phone}")
    return (
        f"15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 "
        f"{phone['energy_pred_j']} J of {phone['energy_budget_j']} J budget, measured "
        f"{phone.get('energy_meas_j')} J"
    )


def check_p9(gate: Gate) -> str:
    """P9: hetero_faulty6 with ``tesar`` -> correct, policy parameters in the report,
    replicas used; the offline MILP runs on the run's own snapshot.json."""
    with tempfile.TemporaryDirectory() as tmp:
        report, events = run_cluster_job(
            Path(tmp),
            "configs/cluster_profiles/hetero_faulty6.yaml",
            6,
            [
                f"job.image_path={ROOT / 'data/images/test_1024.png'}",
                "job.blur_iterations=6",
                "scheduler.policy=tesar",
                "chunking.target_task_s=0.1",
                *FAST_LIVENESS,
            ],
        )
        milp = run(
            py(
                "-m",
                "mdc.coordinator.policies.milp_offline",
                "--snapshot",
                str(Path(tmp) / "out" / "snapshot.json"),
                "--max-tasks",
                "24",
            ),
            timeout=300,
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    params = {
        k: report.get(k)
        for k in ("policy", "alpha", "kappa", "replication", "reliability_target", "max_replicas")
    }
    if params["policy"] != "tesar" or None in params.values():
        raise StepFailed(f"policy parameters missing from report.json: {params}")
    replicas = [e for e in events if e["type"] == "attempt_sent" and e["purpose"] == "replica"]
    if not replicas:
        raise StepFailed("tesar sent no replica although new workers start at rep 0.5")
    expect_ok(milp, "milp_offline on the run's snapshot.json")
    gap = next((ln for ln in milp.stdout.splitlines() if ln.startswith("greedy gap")), "?")
    return (
        f"hetero_faulty6/tesar: output_correct, {report['attempts_total']} attempts, "
        f"{len(replicas)} replicas, makespan {report['makespan_s']} s; MILP {gap}"
    )


def check_p10(gate: Gate) -> str:
    """P10: hetero_malicious6 (30 % and 5 % liars) -> quarantine event, output correct.

    ``verification.base_prob=1.0`` (verify everything): with the default 5 % a rare
    wrong result may legitimately slip through — a measured outcome for experiment
    E5, not something a gate can assert."""
    with tempfile.TemporaryDirectory() as tmp:
        report, events = run_cluster_job(
            Path(tmp),
            "configs/cluster_profiles/hetero_malicious6.yaml",
            6,
            [
                f"job.image_path={ROOT / 'data/images/test_1024.png'}",
                "job.base_tile=64",
                "job.blur_iterations=2",
                "scheduler.policy=tesar",
                "verification.base_prob=1.0",
                *FAST_LIVENESS,
            ],
        )
    if report.get("output_correct") is not True:
        raise StepFailed(f"output_correct={report.get('output_correct')}")
    quarantined = [e for e in events if e["type"] == "worker_quarantined"]
    if not quarantined:
        raise StepFailed("no worker_quarantined event")
    names = report.get("quarantined")
    if not isinstance(names, list) or "bad-phone" not in names:
        raise StepFailed(f"bad-phone (30 % liar) not quarantined: {names}")
    honest = {"fast-phone", "mid-phone", "slow-phone", "laptop-b"}
    if honest & set(names):
        raise StepFailed(f"an honest worker was quarantined: {names}")
    return (
        f"hetero_malicious6/tesar: output_correct, quarantined {names}, "
        f"{report['verifications']} verifications ({report['verification_overhead_pct']} % "
        f"overhead), {report['verification_mismatches']} mismatches, "
        f"{report['repaired_tasks']} tasks repaired, detection {report['detection_latency_s']}"
    )


def check_p11(gate: Gate) -> str:
    """P11: the coordinator CLI serves the dashboard: /healthz public, API refused without
    a session, login with the admin token file, security headers, login URL printed."""
    import http.client  # noqa: PLC0415
    from http import HTTPStatus  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        state, port = Path(tmp) / "state", _free_port()
        base = f"http://127.0.0.1:{port}"
        env = {**os.environ, "HOME": tmp, "PYTHONPATH": str(ROOT)}
        sets = [
            f"security.state_dir={state}",
            f"network.tcp_port={_free_port()}",
            "network.coordinator_bind=127.0.0.1",
            "network.beacon_modes=[]",
            f"dashboard.port={port}",
        ]
        coord = subprocess.Popen(
            py(
                "-m",
                "mdc.coordinator",
                "--output-dir",
                str(Path(tmp) / "out"),
                *[a for kv in sets for a in ("--set", kv)],
            ),
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            if coord.stdout is None:
                raise StepFailed("no coordinator output pipe")
            banner = []
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                line = coord.stdout.readline().decode(errors="replace")
                banner.append(line)
                if "Admin token file" in line or not line:
                    break
            if not any(f"{base}/login" in ln for ln in banner):
                raise StepFailed(f"no dashboard login URL printed: {''.join(banner)[-500:]}")

            def call(
                path: str, data: bytes | None = None, hdrs: dict[str, str] | None = None
            ) -> tuple[int, dict[str, str], bytes]:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    conn.request("POST" if data is not None else "GET", path, data, hdrs or {})
                    r = conn.getresponse()
                    return r.status, dict(r.getheaders()), r.read()
                finally:
                    conn.close()

            status, headers, _ = call("/healthz")
            if status != HTTPStatus.OK or "default-src 'self'" not in headers.get(
                "Content-Security-Policy", ""
            ):
                raise StepFailed(f"/healthz {status} or CSP missing")
            if call("/api/state")[0] != HTTPStatus.UNAUTHORIZED:
                raise StepFailed("/api/state answered without a session")
            token = (state / "admin_token").read_text().strip()
            status, headers, _ = call(
                "/login",
                json.dumps({"token": token}).encode(),
                {"Content-Type": "application/json", "Origin": base},
            )
            cookie = headers.get("Set-Cookie", "")
            if (
                status != HTTPStatus.OK
                or "HttpOnly" not in cookie
                or "SameSite=Strict" not in cookie
            ):
                raise StepFailed(f"login failed or weak cookie: {status} {cookie}")
            session = cookie.split(";", 1)[0]
            status, _, body = call("/api/state", hdrs={"Cookie": session})
            workers = json.loads(body).get("workers") if status == HTTPStatus.OK else None
            if workers is None:
                raise StepFailed(f"/api/state with session: {status}")
        finally:
            coord.terminate()
            try:
                coord.wait(20)
            except subprocess.TimeoutExpired:
                coord.kill()
                coord.wait()
    return (
        f"dashboard served on {base}: login, session cookie, CSP, {len(workers)} worker(s) in state"
    )


def check_p12(gate: Gate) -> str:
    """P12: dist/mdc-worker.zip builds (no secrets, hash file), the unpacked worker runs
    --diag without coordinator code, and the coordinator --diag prints no secrets."""
    import zipfile  # noqa: PLC0415

    sys.path.insert(0, str(ROOT / "scripts"))
    import package_worker  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        try:
            zip_path, digest = package_worker.build(Path(tmp) / "dist")
        except ValueError as exc:
            raise StepFailed(str(exc)) from None
        names = zipfile.ZipFile(zip_path).namelist()
        if any(n.startswith(("mdc/coordinator", "mdc/dashboard")) for n in names):
            raise StepFailed("coordinator/dashboard code in the worker zip")
        phone = Path(tmp) / "phone"
        zipfile.ZipFile(zip_path).extractall(phone)
        env = {**os.environ, "HOME": tmp, "PYTHONPATH": ""}
        diag = run(py("-m", "mdc.worker", "--diag", "--battery", "mock"), env=env, cwd=phone)
        expect_ok(diag, "worker --diag from the unpacked zip")
        state = Path(tmp) / "state"
        coord = run(
            py(
                "-m",
                "mdc.coordinator",
                "--diag",
                "--set",
                f"security.state_dir={state}",
                "--set",
                "network.beacon_modes=[]",
                "--output-dir",
                str(Path(tmp) / "out"),
            ),
            env={**os.environ, "HOME": tmp, "PYTHONPATH": str(ROOT)},
        )
        expect_ok(coord, "coordinator --diag")
        if "MDC1." in coord.stdout or "cert fp:" not in coord.stdout:
            raise StepFailed("coordinator --diag output unexpected")
        size = zip_path.stat().st_size
    return f"worker zip {len(names)} files, {size // 1024} KiB, sha256 {digest[:12]}…; diags ok"


def check_p13(gate: Gate) -> str:
    """P13: run_matrix on e1_baselines (first 10 runs = 2 repetitions of 5 policies) and
    plots.py -> one CSV row per run, both E1 figures, confidence intervals in the summary."""
    import csv  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "runs"
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        expect_ok(
            run(
                py(
                    str(ROOT / "experiments" / "run_matrix.py"),
                    str(ROOT / "configs" / "experiments" / "e1_baselines.yaml"),
                    "--only",
                    "10",
                    "--runs-root",
                    str(root),
                ),
                timeout=1800,
                env=env,
            ),
            "run_matrix e1_baselines --only 10",
        )
        exp = root / "e1_baselines"
        expect_ok(run(py(str(ROOT / "experiments" / "plots.py"), str(exp)), env=env), "plots.py")
        rows = list(csv.DictReader((exp / "results.csv").read_text().splitlines()))
        if len(rows) != 10:  # noqa: PLR2004
            raise StepFailed(f"results.csv has {len(rows)} rows, expected 10")
        failed = [r["point_id"] for r in rows if r["status"] != "ok"]
        if failed:
            raise StepFailed(f"runs not ok: {failed}")
        for fig in ("e1_makespan_by_policy", "e1_failures_by_policy"):
            for ext in ("png", "pdf"):
                if not (exp / "figures" / f"{fig}.{ext}").exists():
                    raise StepFailed(f"missing figure {fig}.{ext}")
        summary = list(csv.DictReader((exp / "figures" / "summary.csv").read_text().splitlines()))
        if not summary or any(not r["makespan_s_ci95"] for r in summary):
            raise StepFailed("summary.csv lacks confidence intervals")
        means = {r["point_id"].split("-")[-1]: float(r["makespan_s_mean"]) for r in summary}
    shown = ", ".join(f"{k} {v:.2f}" for k, v in sorted(means.items(), key=lambda kv: kv[1]))
    return f"e1 x10 runs ok, figures + CIs; mean makespan s: {shown}"


def check_p14(gate: Gate) -> str:
    """P14: the README laptop-only quick start works in a clean checkout.

    Copies exactly what git would ship (tracked + untracked, minus ignored files such as
    .venv and runs/) into a temporary directory with a fresh HOME, then runs the README's
    three commands there (coordinator headless and on free ports so it cannot collide with
    a running instance) and requires a correct output image."""
    listing = run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    expect_ok(listing, "git ls-files")
    with tempfile.TemporaryDirectory() as tmp:
        clone, home = Path(tmp) / "tesar-live", Path(tmp) / "home"
        home.mkdir()
        for rel in filter(None, listing.stdout.split("\0")):
            src = ROOT / rel
            if src.is_file():
                dst = clone / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        env = {**os.environ, "HOME": str(home), "PYTHONPATH": str(clone)}
        expect_ok(run(py("scripts/make_test_images.py"), env=env, cwd=clone), "make_test_images")
        port = _free_port()
        coord = subprocess.Popen(
            py(
                "-m",
                "mdc.coordinator",
                "--run-job",
                "--wait-workers",
                "5",
                "--set",
                "network.max_connections_per_ip=8",
                "--headless",
                "--exit-after-job",
                "--set",
                f"network.tcp_port={port}",
                "--set",
                "network.coordinator_bind=127.0.0.1",
                "--set",
                "network.beacon_modes=[]",
            ),
            cwd=clone,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        cluster = None
        try:
            time.sleep(1.0)
            cluster = subprocess.Popen(
                py(
                    "scripts/local_cluster.py",
                    "configs/cluster_profiles/hetero5.yaml",
                    "--coordinator",
                    f"127.0.0.1:{port}",
                    "--duration",
                    "300",
                ),
                cwd=clone,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            output, _ = coord.communicate(timeout=400)
        finally:
            for proc in (coord, cluster):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        report_path = clone / "runs" / "latest" / "report.json"
        if coord.returncode != 0 or not report_path.exists():
            tail = output.decode(errors="replace")[-600:]
            raise StepFailed(f"quick start failed (exit {coord.returncode}): {tail}")
        report = json.loads(report_path.read_text())
        if report.get("output_correct") is not True:
            raise StepFailed(f"quick start output_correct={report.get('output_correct')}")
        files = sum(1 for _ in clone.rglob("*") if _.is_file())
    return (
        f"README quick start in a clean checkout ({files} files): output correct, "
        f"{report['n_workers_remote']} workers, policy {report['policy']}, "
        f"makespan {report['makespan_s']} s"
    )


PHASE_CHECKS: dict[int, Callable[[Gate], str]] = {
    0: check_p0,
    1: check_p1,
    2: check_p2,
    3: check_p3,
    4: check_p4,
    5: check_p5,
    6: check_p6,
    7: check_p7,
    8: check_p8,
    9: check_p9,
    10: check_p10,
    11: check_p11,
    12: check_p12,
    13: check_p13,
    14: check_p14,
}


def step_phase_checks(gate: Gate) -> str:
    """Run PHASE_CHECKS for every phase <= N (earlier ones as regression)."""
    notes = []
    for phase in sorted(p for p in PHASE_CHECKS if p <= gate.phase):
        notes.append(f"P{phase}: {PHASE_CHECKS[phase](gate)}")
    missing = [p for p in range(gate.phase + 1) if p not in PHASE_CHECKS]
    if missing:
        raise StepFailed(f"PHASE_CHECKS missing for phase(s) {missing}")
    return " | ".join(notes)


STEPS: list[tuple[str, Callable[[Gate], str]]] = [
    ("1 environment", step_environment),
    ("2 compile", step_compile),
    ("3 format", step_format),
    ("4 lint", step_lint),
    ("5 types", step_types),
    ("6 sast", step_sast),
    ("7 forbidden", step_forbidden),
    ("8 isolation", step_isolation),
    ("9 dep-audit", step_audit),
    ("10 unit+prop+integ", step_unit),
    ("11 security", step_security),
    ("12 e2e", step_e2e),
    ("13 perf/soak", step_perf_soak),
    ("14 critical-cov", step_critical_coverage),
    ("15 skip-guard", step_skip_guard),
    ("16 phase-checks", step_phase_checks),
]


def report(gate: Gate, failed_step: str | None) -> None:
    """Print the summary table, write the JSON report, append to PROGRESS.md."""
    width = max(len(r.name) for r in gate.results) if gate.results else 10
    print("\n" + "=" * 100)
    print(f"{'step':<{width}}  {'status':<6}  {'time':>7}  notes")
    print("-" * 100)
    for r in gate.results:
        print(f"{r.name:<{width}}  {r.status:<6}  {r.duration_s:6.1f}s  {r.notes}")
    if gate.exemptions:
        print("\nReviewed forbidden-pattern exemptions:")
        for e in gate.exemptions:
            print(f"  - {e}")
    print("=" * 100)
    verdict = (
        f"PHASE {gate.phase} GATE: PASS"
        if failed_step is None
        else f"PHASE {gate.phase} GATE: FAIL (step {failed_step})"
    )
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    (GATES_DIR / f"phase-{gate.phase}.json").write_text(
        json.dumps(
            {
                "phase": gate.phase,
                "time": stamp,
                "verdict": verdict,
                "offline": gate.offline,
                "steps": [r.__dict__ for r in gate.results],
                "exemptions": gate.exemptions,
            },
            indent=2,
        )
    )
    progress = ROOT / "PROGRESS.md"
    if progress.exists():
        lines = [f"\n### Gate run — phase {gate.phase} — {stamp}", "", f"**{verdict}**", ""]
        lines += [f"- {r.name}: {r.status} — {r.notes}" for r in gate.results]
        if gate.offline:
            lines.append("- NOTE: run with --offline; pip-audit must be run later.")
        with progress.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print(verdict)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the gate."""
    parser = argparse.ArgumentParser(description="TESAR-Live phase gate (spec G.2)")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--repeat-e2e", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--soak", action="store_true", help="also run the soak test")
    args = parser.parse_args(argv)
    if not 0 <= args.phase <= LAST_PHASE:
        parser.error("--phase must be in 0..14")

    os.chdir(ROOT)
    gate = Gate(args.phase, max(1, args.repeat_e2e), args.offline, args.keep_going, args.soak)
    failed: str | None = None
    for name, func in STEPS:
        print(f"\n--- step {name} ---", flush=True)
        t0 = time.monotonic()
        try:
            notes = func(gate)
            status = "PASS"
        except StepSkipped as exc:
            status, notes = "SKIP", str(exc)
        except StepFailed as exc:
            status, notes = "FAIL", str(exc)
        except subprocess.TimeoutExpired as exc:
            status, notes = "FAIL", f"timed out after {exc.timeout}s"
        except Exception as exc:  # any crash = failed step
            status, notes = "FAIL", f"{type(exc).__name__}: {exc}"
        gate.results.append(StepResult(name, status, time.monotonic() - t0, notes))
        if status == "FAIL":
            failed = failed or name.split()[0]
            if not gate.keep_going:
                break
    report(gate, failed)
    return 0 if failed is None else 1


if __name__ == "__main__":
    sys.exit(main())
