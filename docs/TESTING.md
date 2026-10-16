# Testing and the phase gate

Every phase ends with `python scripts/phase_gate.py --phase N` printing `PHASE N GATE: PASS`.
The gate is also run by CI (`.github/workflows/ci.yml`) for the phase stored in `.phase`.

## Gate steps (spec G.2)

| # | Step | Tool |
|---|---|---|
| 1 | Environment: Python, tools present, no stray `*.join`/`*.pem`/`*.key` files | — |
| 2 | Compile everything | `python -m compileall` |
| 3 | Formatting | `ruff format --check .` |
| 4 | Lint (`E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL`) | `ruff check .` |
| 5 | Types (`disallow_untyped_defs`; `strict` for security-critical packages) | `mypy mdc` |
| 6 | SAST (medium/high fail) | `bandit -r mdc scripts -ll` |
| 7 | Forbidden patterns (G.4) | `scripts/check_forbidden.py` |
| 8 | Worker isolation (worker side = stdlib + numpy) | `scripts/check_worker_isolation.py` |
| 9 | Dependency audit of the hash-pinned locks | `pip-audit` (skipped with `--offline`) |
| 10 | Unit + property + integration tests with branch coverage ≥ T(N) | `pytest --cov` |
| 11 | Security suite | `pytest -m security` |
| 12 | End-to-end (Phase ≥ 4; ×3 at milestones 5, 10, 14) | `pytest -m e2e` |
| 13 | Perf smoke (Phase ≥ 9) and soak (Phase 14 / `--soak`) | `pytest -m perf` / `-m soak` |
| 14 | ≥ 95 % line coverage of security-critical modules | coverage JSON |
| 15 | Skip/regression guard: no new skips, no vanished tests | `runs/gates/baseline.json` |
| 16 | Phase-specific checks for every phase ≤ N | `PHASE_CHECKS` in the gate |
| 17 | Report: table, `runs/gates/phase-N.json`, summary appended to `PROGRESS.md` | — |

Coverage thresholds T(N): phases 0–3 → 70 %, 4–7 → 80 %, 8–14 → 85 %.

Options: `--repeat-e2e K`, `--offline` (skips pip-audit, noted in PROGRESS.md),
`--keep-going` (run all steps even after a failure), `--soak`.

## Test layers and markers

| Layer | Location | Marker |
|---|---|---|
| Unit | `tests/test_*.py` | (none) |
| Property / fuzz | `tests/property/` | `property` |
| Integration (loopback sockets) | `tests/test_*.py` | `integration` |
| End-to-end (subprocesses) | `tests/test_e2e_*.py` | `e2e` |
| Security | `tests/security/` | `security` (every test there) |
| Performance smoke | `tests/test_perf_smoke.py` | `perf` |
| Soak | `tests/test_soak.py` | `soak` |

Markers are strict (`--strict-markers`): a typo in a marker name is an error. The default
per-test timeout is 60 s. Warnings raised from `mdc` and leaked resources (`ResourceWarning`)
are errors, and an autouse fixture fails any test during which asyncio reported a destroyed
pending task.

## How to add a test

1. Put it next to the component it tests (`tests/test_<component>.py`), or in
   `tests/security/test_sec_<area>.py` with `pytestmark = pytest.mark.security`.
2. Never test only the happy path: include invalid input, boundary values and failure cases.
3. Use the fixtures from `tests/conftest.py` — they clean up even when a test fails:
   `tmp_state_dir`, `tls_cert`, `security_log`, `registry`, `enrolled(name, port=...)`,
   `worker_proc(join_file, *flags)` (subprocess in its own process group, log captured, killed
   at teardown); from Phases 4–5 `coordinator(...)` and `cluster(profile)`. Helpers:
   `tests/helpers/stub_coordinator.py` (a TLS coordinator that authenticates workers and lets a
   test drive the session), `tests/helpers/worker_harness.py` (in-process worker),
   `tests/helpers/procs.child_env()` (environment for any subprocess a test starts — tests must
   not rely on `pip install -e .`).
4. Never delete or deselect an earlier test to make the gate pass; the skip guard detects it.
5. If a test is genuinely wrong, fix it and record why in `DECISIONS.md`.

## Phase checks (`PHASE_CHECKS`, step 16)

Each phase adds one functional check that runs real processes. At phase N the checks for every
phase ≤ N run as a regression suite:

| Phase | Check |
|---|---|
| 0–3 | config and state-dir permissions; certificate, enrollment and pinning; tiled = whole-image reference; worker `--diag` prints no secret |
| 4–6 | headless jobs with local workers; crash + drop recovery; `hetero5` with `fastest`, probes and new event fields |
| 7–10 | `hetero_faulty6` reputations; 15 % battery worker idle; `tesar` with replicas + MILP on the run's snapshot; `hetero_malicious6` quarantine with correct output |
| 11 | CLI-served dashboard: public healthz, 401 without session, token login, strict cookie, CSP |
| 12 | worker zip (no secrets, hash), unpacked worker runs `--diag` without coordinator code, coordinator `--diag` clean |
| 13 | 10 real E1 runs → CSV rows, figures, confidence intervals |
| 14 | README laptop-only quick start in a clean checkout (fresh HOME) → correct output |

Further helpers: `tests/helpers/dashboard.py` (fake dashboard backend and a test client);
`tests/test_code_quality.py` enforces module docstrings and typed public functions.
