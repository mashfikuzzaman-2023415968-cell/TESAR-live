# Progress

## Phase 0 — Project skeleton (2026-10-07)

**Built**
- Full directory tree from spec Section 3; not-yet-built modules are docstring placeholders
  naming the phase that implements them.
- `pyproject.toml` (package metadata; pytest markers/timeouts/strict markers/warnings-as-errors;
  ruff, mypy, coverage, bandit config), `requirements*.txt`, hash-pinned `requirements.lock` and
  `requirements-worker.lock`, `.gitignore`, `.pre-commit-config.yaml`, CI workflow, `.phase`.
- `configs/default.yaml` (verbatim from spec Section 5) and `mdc/common/config.py`: typed
  dataclasses, deep merge, `--set` overrides, strict validation (unknown keys / wrong types /
  ranges), insecure-combination refusal.
- `mdc/common/`: `ewma.py`, `events.py` (JSONL + subscribers + recent ring buffer),
  `logging_setup.py` (console + 5 MB × 3 rotating file), `ids.py` (`secrets`-based),
  `timeutil.py`, `statedir.py` (0700 dir, atomic 0600 secret files, permission refusal).
- Tooling: `scripts/phase_gate.py` (all 17 steps of G.2), `scripts/check_forbidden.py` (G.4),
  `scripts/check_worker_isolation.py`, `scripts/make_test_images.py` (4 deterministic images).
- Docs: README, SECURITY.md (threat model summary + residual risks), docs/SECURITY_DESIGN.md
  (Phase-0 controls), docs/TESTING.md, OBSERVATIONS.md template, DECISIONS.md, doc stubs.

**Tested**
- `tests/test_config.py`: defaults == default.yaml, merge, overrides, unknown keys, wrong types,
  ranges and boundaries, bad files.
- `tests/test_common.py`: EWMA (formula, bounds, hostile values), ids, events (JSONL format,
  sanitising, log-injection escaping, subscribers, thread safety), logging rotation, worker-side
  modules do not import yaml.
- `tests/security/test_sec_config.py`: insecure combos, state-dir/secret-file permissions,
  atomic writes.
- `tests/security/test_sec_static.py`: repository is clean under check_forbidden, bandit and the
  isolation check; every scanner rule fires on a bad sample and stays quiet on safe code.

**Known issues / notes**
- Docs other than TESTING/SECURITY are stubs, filled in by their phases.


## Phase 1 — Protocol and authentication (2026-10-07)

**Built**
- `common/protocol.py`: MDC1 framing (prefix checked before any body read), strict JSON headers
  (no NaN/Infinity/overflow/huge ints/duplicate keys/depth > 8), all 13 message types,
  `read_frame`/`write_frame` (per-connection lock), `encode_frame`/`decode_frame`,
  `pack_array`/`unpack_array` (little-endian, zlib, bounded decompression, SHA-256 transport
  integrity), error hierarchy `WireError` → `ProtocolError`/`ValidationError`/`IntegrityError`/
  `ConnectionClosed`.
- `common/validation.py`: one validator per message type, payload rules, task-kind whitelist,
  parameter limits, input-shape checks, per-kind result validators, clamping of self-reported values.
- `common/auth.py`: mutual HMAC-SHA256 with length-prefixed fields, domain-separation labels and
  certificate-fingerprint channel binding; constant-time verification.
- `common/tls.py`: hardened server/client contexts, exact-certificate pinning,
  `open_pinned_connection` (closes before any application byte on mismatch).
- `common/credentials.py` (join codes, 0600 join files, env var), `common/redact.py` (key/value
  redaction, record-factory + handler filter), `common/safe_names.py`.
- `coordinator/certs.py` (ECDSA P-256 self-signed, rotation, expiry/mismatch/permission checks),
  `coordinator/enrollment.py` (locked atomic registry, enroll/revoke/rotate/list/lookup/reload),
  `coordinator/security_log.py` (whitelisted, redacted, sanitized, rotated 0600, rate-limited),
  `beacon.detect_lan_ip`.
- `mdc.cli`: `enroll [--out] [--qr]`, `revoke`, `rotate-secret`, `list-devices`,
  `show-fingerprint`, `rotate-cert --yes [--reissue-dir]`.
- Log files are created 0600 (`PrivateRotatingFileHandler`).
- Docs: `docs/PROTOCOL.md` (complete), `docs/SECURITY_DESIGN.md` (Phase-1 controls).
- Gate: `PHASE_CHECKS[1]` (temp cert, enrollment, join-code round trip, TLS pin ok / wrong pin refused).

**Tested** — 708 tests in total (unit, property/fuzz, integration, security), 99.4 % branch coverage
- `test_protocol.py`: round trip of every type (bytes + stream), 10 MB via StreamReader and real
  TCP, byte-by-byte delivery, concurrent writers, every limit and malformed header, arrays of
  every dtype/shape/compression, tampering, decompression bomb with `tracemalloc` bound.
- `test_auth.py`: spec construction, wrong secret/id/nonces/fingerprint, domain separation,
  unambiguous encoding, malformed inputs.
- Security suite: `test_sec_tls.py` (pin ok, wrong pin before app data, impostor cert, plaintext
  client, TLS < 1.2, handshake timeout), `test_sec_auth.py` (full handshake over TLS; unknown,
  revoked, wrong, rotated credentials; rogue coordinator; replay; reflection; MITM fingerprint),
  `test_sec_enrollment.py` (join codes, registry, locking, reload, certificates, CLI),
  `test_sec_validation.py`, `test_sec_secrets.py`.
- Fuzzing (`tests/property/`): decoders/validators/join-code parser raise only declared errors,
  bounded memory.

**Found and fixed while testing**
- CLI leaked the security-log file handle (caught by `ResourceWarning` → error).
- pytest's own log handler received unredacted join codes → records are now redacted at creation.
- Log files were created with the umask (0664) → now always 0600.
- `bytes.fromhex` accepts whitespace → nonces are matched with a strict regex first.

- Verified on a real Python 3.10 interpreter with the hash-locked dependencies (708/708, ×3);
  this exposed a 3.10-only socket leak in a test server (fixed; see DECISIONS.md).

**Known issues / notes**
- CI (`.github/workflows/ci.yml`) has not run yet: the repository has no GitHub remote.
- The handshake *drivers* (worker loop, coordinator server) are Phases 3–4; Phase 1 provides the
  building blocks and a reference handshake in `tests/helpers/handshake.py`.


## Phase 2 — Compute tasks, tiling, DAG (2026-10-07)

**Built**
- `mdc/tasks/image_pipeline.py`: integer box blur (summed-area table, valid convolution), Sobel L1
  magnitude, `pipeline_valid`, edge-replicated `reference`; all limits enforced inside the kernels;
  cooperative deadline.
- `mdc/tasks/tiling.py`: `TileGrid` (rows/cols, chunk rectangles, halo input slices, split,
  row strips, assembly), `chunk_runs`.
- `mdc/tasks/row_stats.py`, `mdc/tasks/synthetic.py` (hash chain + XOR aggregation),
  `mdc/tasks/base.py` (`TASK_KINDS` registry with validated inputs/outputs, `execute`, cost model),
  `mdc/tasks/benchmark.py` (ms per work unit), `mdc/tasks/errors.py`.
- `mdc/coordinator/job.py`: `TaskNode`, `Attempt`, `Job` (readiness and scheduling order,
  idempotent `mark_done`, `invalidate` cascades with epochs, `replace_result`, `requeue`, chunk
  inputs, stage-2 strips, `run_final`), `execute_inprocess`.
- `mdc/coordinator/job_builder.py`: safe image loading/saving, image and synthetic job builders.
- `mdc.cli reference | compare | bench`.
- Gate: `PHASE_CHECKS[2]` runs `mdc.cli reference` on `test_1000x1300.png` and checks tiled
  assembly (chunks of 1, 3, 8, 2 tiles) is byte-identical.

**Tested**
- Kernels vs naive double-loop implementations; constant images; saturation; shapes; overflow;
  limits; deadlines; pinned regression digest.
- Tiling exactness: 40 seeded cases + 250 Hypothesis examples (random sizes incl. non-multiples,
  r ∈ {1,2,3}, iters ∈ {1..4}, random chunkings) — always byte-identical to the reference.
- DAG: readiness, order, idempotency, invalidation cascades (row stats and final), partial
  dependencies, epochs reject stale attempts, replicas/requeue, chunk validation, synthetic jobs.
- Security: only whitelisted kinds run; hostile params/inputs refused; worker's own stricter limits
  win; deadlines stop kernels.

**Manual check**
- `python -m mdc.cli reference --image data/images/test_512.png --r 2 --iters 4 --out ref.png`
  produces a clean edge image (shape outlines, background noise blurred away).
- `python -m mdc.cli bench` → ≈ 3.8 ms per work unit on this laptop.
- Full suite also passes on Python 3.10.22 (913/913).


## Phase 3 — Worker node and discovery (2026-10-07)

**Built**
- `mdc/worker/worker.py`: `Worker` (target resolution, pinned TLS, mutual handshake, reconnect
  with exponential backoff 1..15 s keeping the same `worker_id`) and `Session` (reader loop that
  never computes, PING→PONG, PROBE→PROBE_ACK, TASK_ASSIGN with bounded queue / battery floor /
  integrity and limit checks, TASK_CANCEL for queued and running tasks, CONTROL gating, BYE,
  unknown-message tolerance up to 10, idle timeout, battery refresh loop, flaky disconnects).
- `mdc/worker/executor.py` (thread pool, cooperative deadline `--task-max-s`),
  `battery.py` (Termux:API, sysfs, mock; sign-normalized current; single safe subprocess),
  `faults.py` (drop, crash, crash-after, malicious with plausible corruption, slowdown, flaky),
  `shaper.py` (delay + bandwidth for bulk sends only), `discovery_listener.py` (UDP broadcast +
  multicast, fingerprint filter, 1 KB limit, 10/s rate limit).
- `python -m mdc.worker` CLI with all spec flags, lab-mode gating with a red banner, root refusal,
  credentials from a 0600 file / stdin / env only, `--diag`.
- `tests/helpers/stub_coordinator.py` (also runnable for the manual check) and an in-process
  worker harness.
- Gate: `PHASE_CHECKS[3]` (worker `--diag` output contains no secret).

**Tested**
- Integration over TLS loopback (24 tests): authentication, PONGs < 0.5 s while computing, image
  and row-stats results, reconnect after coordinator restart, busy, cancel, integrity error,
  limit errors, time limit, battery floor and charging bypass, shaping, unknown messages, BYE,
  idle timeout, limit tightening only, accepted slots, every fault, CONTROL, revocation exit,
  wrong proof.
- Units: battery parsers (Termux variants, sysfs current/power, mock drain), corruption
  plausibility, shaper math, seeded faults, beacon parsing and real UDP discovery, rate limit.
- Security (`test_sec_worker_safety.py`): unsafe CLI combinations refused, root refusal, loose
  join file refused, pin mismatch sends nothing, unknown kind disconnects, CONTROL ignored and
  logged, battery floor, worker limits hold, spoofed/oversize beacons, one external command only,
  **a real worker process has no listening socket and writes only its 0600 log**, `--diag` leaks
  no secret, SIGTERM stops cleanly.

**Manual check** (`python -m tests.helpers.stub_coordinator` + a real worker using the laptop's
real sysfs battery): PONG round trips 26–32 ms during a 1.57 s task (≈ 1.6 ms idle), correct
result, immediate detection when the stub was killed, then reconnect attempts with backoff.

**Found and fixed while testing**
- SIGTERM hung for 30 s on a connected worker (TLS `close_notify` wait) → abort on stop.
- Termux "UNPLUGGED" with unknown status was treated as maybe-charging → now not charging.
- A test contained a join-code-shaped literal → caught by `check_forbidden.py`, now built at runtime.
- Audit against the spec after the gate found 4 gaps, now fixed: `--local` now requires a loopback
  coordinator; `--name` is validated at the CLI; the G.1 `worker_proc` fixture exists (own process
  group, log capture, killed at teardown — with a test); `docs/SECURITY_DESIGN.md` documents the
  Phase 2 and 3 controls. An observation about PONG latency during compute went to OBSERVATIONS.md.
- While making those fixes, an edit accidentally deleted two tests and renumbered parametrized test
  ids; the gate's skip/regression guard (step 15) caught both. The tests were restored from git and
  new cases appended so earlier ids stay stable.
- On Python 3.10 (no `pip install -e .`), worker subprocesses started by tests could not import
  `mdc` — CI would have failed the same way. Test subprocesses now get an explicit `PYTHONPATH`
  (`tests/helpers/procs.child_env`). Full suite passes on Python 3.10.22 without installation.


## Phase 4 — Coordinator core: first end-to-end milestone (2026-10-07)

**Built**
- `coordinator/server.py`: TLS accept, connection and per-IP limits, auth-failure bans, handshake
  (registry or ephemeral local credential, MAC, trusted-local rules), session replacement on
  reconnect, reader loop with idle timeout and message limits, disconnect handling, live revocation.
- `coordinator/registry.py` (`WorkerRecord`, state machine with `worker_state` events, per-kind
  speed EWMA), `dispatcher.py` (attempts, ownership, unsolicited limit, result validation,
  idempotent acceptance, cancellation of duplicates, traffic counters), `scheduler.py`
  (event-driven + 0.5 s tick, enforces capacity and validity of policy output),
  `policies/` (`base` context and interface, `local`, `round_robin`, registry),
  `failure_detector.py` (ping loop, PONG bookkeeping), `local_worker.py` (supervised subprocess,
  ephemeral credential, restart), `beacon.py` (broadcast + /24 + multicast), `assembler.py`
  (background reference with cache, final stage, outputs, byte comparison), `metrics.py`
  (report.json per 11.3), `coordinator.py` (orchestration, snapshot, job life cycle),
  `python -m mdc.coordinator` CLI.
- `common/ratelimit.py` (token bucket, failure tracker with bans and LRU bound, sliding counter).
- Test infrastructure: `coordinator(...)` fixture, raw protocol client for hostile-worker tests.
- Gate: `PHASE_CHECKS[4]` (CLI coordinator + 2 worker processes, 512 x 512 job, output correct).

**Tested**
- End-to-end (`test_e2e_basic.py`): in-process coordinator + 3 real worker processes,
  `round_robin` (every remote worker did work) and `local`; the coordinator CLI as a subprocess;
  insecure-dev mode; CLI refusals.
- Integration (15): jobs with in-process workers, worker lost mid-job → reassigned, reconnect keeps
  the record, BYE → LEFT, session replacement, local-worker restart, job timeout, every result
  rejection path, compute-time clamping.
- Security: resource limits (10), auth additions (wrong worker, trusted-local spoof, live
  revocation, rotation, unknown/revoked live, late enrollment), rate limiter (100 % coverage).
- Units: policies, scheduler enforcement against an "evil" policy, registry, dispatcher
  predictions, beacon sender, report.

**Manual check** — `mdc.cli enroll` × 2, `python -m mdc.coordinator --headless --run-job
--exit-after-job --wait-workers 2 --set scheduler.policy=round_robin`, one worker *without*
`--coordinator` (found the coordinator by UDP beacon on this machine), one with it, plus a worker
with a tampered join file: job finished with output identical to the reference (`mdc.cli compare`),
the tampered worker got `REJECT auth`, and `security.jsonl` recorded `auth_failed (bad MAC)`.

**Found and fixed while testing**
- Security log: a field named `type` overwrote the event type with a peer-controlled value (audit
  log forgery) → reserved names refused.
- Test helpers treated a timeout as "connection closed" on Python ≥ 3.11 (`TimeoutError` is an
  `OSError` subclass there) → assertions could pass falsely; fixed, then re-verified the server.
- The spec's `job_started.type` field collides with the event type → `job_type`.
- A device enrolled after start had to wait for the 2 s registry poll → re-read on unknown id.
- Python 3.10: test clients whose connection was closed by the *server* leaked their TLS transport
  (3.10 only marks it closed when the owner closes it) → helpers now close their side and never leak
  a half-done handshake. Production code already closes its own side. Full suite: 1117/1117 on
  Python 3.10.22 (×2) and 3.13; e2e repeated 3 × green.

**Known issues / notes**
- Default `scheduler.policy: tesar` is refused until Phase 9 (use `round_robin` / `local`).
- A worker that hangs *without* disconnecting stalls its tasks until Phase 5 adds timeouts.


## Phase 5 — Failure detection and recovery (2026-10-07, milestone)

**Built**
- `coordinator/failure_detector.py`: ping loop + 0.25 s check loop; ACTIVE → SUSPECT → DEAD with
  `suspect_after_s`/`dead_after_s`; PONG restores SUSPECT → ACTIVE; DEAD reassigns in-flight work and
  force-closes the connection; RTT sanity check.
- `dispatcher.py`: per-attempt timeouts (`max(min, factor·predicted + slack)`, swept every 0.25 s,
  TASK_CANCEL sent, requeue unless another copy is in flight); late-result acceptance; reputation
  hooks on every outcome; `on_result` split into ownership / decode / accept.
- `scheduler.py`: trusted local fallback after `max_attempts_per_task`; error backoff eligibility.
- `coordinator/reputation.py`: Phase-7 interface with counting implementation.
- `scripts/local_cluster.py` + the four profiles; `scripts/chaos.py` (kill/stop/cont, seeded
  random kills, clock from `job_started`); `cluster(profile)` fixture.
- Gate: `PHASE_CHECKS[5]` (CLI job with a crashing and a dropping worker → correct).

**Tested**
- E2E with real worker processes (`test_e2e_failures.py`): crash-after (fault really fired, worker
  DEAD), drop-everything (timeouts), **SIGSTOP → SUSPECT → DEAD → SIGCONT → reconnect**, flaky,
  all remote workers failing → all 20 tasks completed by the local fallback, the `hetero_faulty6`
  cluster profile. Every test asserts byte-exact output **and no task completed twice**.
- Units (`test_failure_detection.py`, fake clock/channels): state machine and thresholds, DEAD
  reassignment, timeout sweep, waiting for another copy, the four late-result cases, fallback,
  no-local warning, error backoff growth/cap/reset, reputation stub. `test_cluster_tools.py`:
  profiles, enrollment/re-keying, 0600 join files, launch/stop, chaos parsing/plans/execution.

**Section 12 concurrency checklist (verified against the code, see also Phase 14)**
- [x] No blocking work on the event loop: certificate, benchmark, job build, reference, final
  assembly, PNG encoding and battery reads run in threads. What remains on the loop is bounded:
  packing/unpacking one chunk (≤ a few hundred KB), small JSON/file appends (events, security log,
  `devices.json` last-seen).
- [x] One reader task per connection; all writes go through the connection's `asyncio.Lock`.
- [x] Every network read has a timeout: handshake reads inside `wait_for(handshake_timeout_s)`,
  session reads inside `wait_for(idle_*_timeout_s)` (coordinator and worker).
- [x] All 15 `create_task` sites are tracked and cancelled/awaited on shutdown (scheduler, detector
  ×2, beacon, revocation watcher, handlers, dispatcher sends, local-worker supervisor, worker
  background/task coroutines); the autouse fixture fails any test that leaks a pending task.
- [x] A result is accepted at most once (idempotent `mark_done`, attempt status, epochs) —
  asserted by "no duplicate task_done" in every failure e2e test.
- [x] The job cannot hang while a local worker exists: every in-flight attempt ends by result,
  timeout, loss (DEAD) or cancellation, and exhausted tasks fall back to the local worker.
- [x] Unknown message types logged and ignored (≤ 10); malformed frames close the connection.
- [x] Random choices use seeded `random.Random` instances (`rng`), never the module functions.
- [x] Durations use `time.monotonic()` / `perf_counter()`; no `time.time()` anywhere in `mdc/`.

**Found and fixed while testing**
- Hot retry loop against a worker returning errors → exponential error backoff.
- The development laptop on AC power at 2 % battery ("Not charging") made the local worker refuse
  all work → mains power counts as charging; local worker has no battery floor (OBSERVATIONS.md).
- Two sysfs tests silently depended on the real machine's power state → isolated.
- Local clusters exceeded the per-IP connection limit silently → coordinator warning + launcher hint.
- Full suite 1165/1165 on Python 3.13 and 3.10.22; milestone gate ran e2e ×3 green.


## Phase 6 — Network estimation and adaptive chunking (2026-10-08)

**Built**
- `coordinator/netestimator.py`: RTT EWMA + minimum (seeded by the handshake round trip), bandwidth
  EWMA from the join probe and passive per-attempt samples (noise filters, clamps), per-kind and
  per-direction compression ratios, `predict_transfer_s`, snapshots.
- Bandwidth probe at join: PROBING until PROBE_ACK (or 3 s), `probe_done` events.
- `policies/base.py`: `predicted_s = queue_wait + rtt + wire_bytes / bw + compute` with exact chunk
  geometry; workers' estimates in `WorkerView`, compression table in the context.
- `coordinator/chunking.py`: time-target and loss-tolerance rule, min/max, row boundaries.
- `policies/fastest.py` (pulled forward from Phase 9, see DECISIONS.md).
- Dispatcher: predictions for every attempt, compression and passive-bandwidth feeds,
  `est_rtt_s`/`est_bw_bytes_per_s` in `attempt_sent`; report `avg_rtt_ms`/`bw_kbps`.
- RTT measured from the coordinator's own PING records (echo only checked).
- Gate: `PHASE_CHECKS[6]` (hetero5 cluster with `fastest`, new event fields, probes).

**Tested**
- `test_netestimator.py`: EWMA, minimum, probe and passive formulas on hand-computed numbers, noise
  filters, hostile values (NaN, ±inf, negative, huge, non-numbers) never reach the estimates, clamps,
  compression ratios.
- `test_chunking.py`: time-target formula, fast > slow, low bandwidth → smaller, high λ → smaller and
  the loss tolerance is met, min/max, row boundaries and runnable tiles, non-adaptive fixed size,
  synthetic tasks unchunked, shared halo.
- `test_policies.py`: `fastest` picks the fast device, counts queue wait and bandwidth, fills
  capacity without overlaps, respects `allow_local` and reserved tasks, bigger chunks for fast workers.
- Integration: probe → ACTIVE with a bandwidth sample; unanswered probe → ACTIVE after 3 s; foreign
  probe id ignored; PINGs feed RTT; shaped link gets a lower estimate; event fields present.
- E2E (`test_e2e_chunking.py`, real processes): the fast worker's chunks are bigger than the slow
  worker's, and its measured speed is faster.

**Manual check (hetero5)** — `slow-phone` gets smaller chunks only when per-tile costs matter: with
the default `target_task_s = 2 s` everything on this laptop earns the maximum chunk (see DECISIONS.md).

**Found and fixed while testing**
- Raw test clients never read the 256 KiB probe → paused reading → graceful TLS close could not
  finish → leaked socket (order-dependent failure). Clients now ACK probes and abort on close.
- Tests that pinned the old prediction (compute + 0.5 s) and the old echoed-`t_c` RTT were updated.
- The bandwidth estimate's asymmetry/contamination limits were measured and documented.
- **Python 3.10 shutdown hang** (asyncio `wait_for` swallowing cancellation, CPython gh-86296):
  diagnosed with a watchdog dump, fixed with a cancellation-safe `wait_event`, regression test proven
  to fail on the old code under 3.10 (DECISIONS.md).
- Worker start-up race: "starting" was logged before the SIGINT/SIGTERM handlers existed, so a
  signal right after it could kill the worker uncleanly (-15) — now logged once handlers are installed.
- Full suite 1208/1208 on Python 3.13 and 3.10.22 (3.10 run: 4 m 46 s, previously 24 min with hangs).
- The SIGSTOP e2e test assumed recovery via DEAD; with better predictions the attempt timeout can
  fire first — both are correct, the test now accepts either (DECISIONS.md).

## Phase 7 — Reputation and reliability model (2026-10-15)

**Built**
- `coordinator/reputation.py`: Beta reputation per `worker_id` (priors, decay, per-kind failure
  weights; `wrong_result` weighs 5×), failure rate λ from EWMAs of failure fraction and attempt length
  (Poisson inversion, DECISIONS.md), `R = rep · exp(-λ · predicted_s)` with trusted 0.9999 and
  battery-ineligible 0, snapshot (`rep`, `lambda`, `s`, `f`, counts).
- Hooks: every success, timeout, disconnect and error path of the dispatcher records the event and
  the attempt; the wrong-worker case records an error for the sender.
- Scheduler: `WorkerView.rep`, `failure_rate_per_s` (measured λ) and `eligible`;
  `SchedulingContext.reliability(chunk, worker)` for policies; chunking now sees the real λ.
- `attempt_sent.predicted_R`; `report.json` `per_worker.rep_final`/`lambda_final`.
- Gate: `PHASE_CHECKS[7]` (hetero_faulty6 cluster: correct output, `predicted_R` on every attempt,
  the unreliable devices end below every stable one).

**Tested** (`test_reputation.py`)
- New worker 0.5; successes raise it; one wrong result drops it > 3× more than a timeout; decay makes
  old failures matter less; counts and unknown kinds; config weights used.
- λ: default until 3 attempts, increases with failures, matches the Poisson model, floor 1e-6.
- R decreases with predicted time; trusted / ineligible special cases; battery eligibility.
- Hostile values (NaN, ±inf, negative durations) never break λ or R.
- Hypothesis: rep always in (0, 1); a success never lowers and a failure never raises it; λ > 0 and
  R ∈ [0, 1] for arbitrary attempt histories.

**Manual check (hetero_faulty6, round_robin, 1024×1024)** — rep_final: laptop-b 0.96, local 0.95,
fast-phone 0.94, crasher-phone 0.94 (the job ended before its 15 s crash), slow-phone 0.86,
flaky-phone 0.83, dropper-phone 0.42.

**Found and fixed while testing**
- Two e2e tests had been tuned to a throttled CPU and failed at full clock speed even on the
  Phase 6 code: the chunking test now calibrates its workload from the benchmark; the crash test
  uses a longer job and additionally asserts that the crash hit in-flight work (DECISIONS.md).
- `--slowdown` did not slow the reported benchmark, so emulated slow phones got fast-device first
  chunks; fixed and tested.
- Lesson: never run the 3.10 suite in parallel with a gate — timing tests share the CPU.
- Full suite 1228/1228 on Python 3.13 (gate) and 3.10.22 (2 m 15 s).

## Phase 8 — Energy model and battery measurement (2026-10-15)

**Built**
- `coordinator/energy.py`: `E = p_compute·compute_s + p_tx·transfer_s` per device class, battery
  capacity and per-job budget, per-worker ledger (committed, spent, measured, idle baseline),
  feasibility (battery floor or charging, budget; local exempt), integration of PONG current ×
  voltage samples with clamps.
- `coordinator/estimators.py`: the run's network, reputation and energy models in one object.
- Dispatcher: predicted energy per attempt (`attempt_sent.predicted_energy_j`), commit on send,
  settle from measured times on success / prediction on failure; `WorkerRecord.energy_spent_j`.
- Failure detector: battery samples from every PONG feed measured energy.
- Scheduler: battery and budget gate for remote workers (events `worker_ineligible_battery`,
  `energy_budget_exhausted`), `WorkerView.energy_left_j`, per-assignment enforcement;
  `SchedulingContext.energy_j / energy_feasible`; `fastest` and `round_robin` skip unaffordable
  workers.
- Report per worker: `energy_pred_j`, `energy_meas_j`, `energy_budget_j`, `idle_baseline_w`,
  battery start/end, `charging`.
- Gate: `PHASE_CHECKS[8]` (worker at 15 % gets nothing and is logged; output correct; energy fields).

**Tested** (`test_energy.py`)
- Energy formula, budget (charging/unknown/disabled → unlimited, clamped %), feasibility (over
  budget, low battery, charging bypass, local exemption), ledger commit/settle, budget fixed per job
  and reset by `start_job`, current integration (idle vs job, gap cap, charging and missing values
  skipped), hostile battery values bounded (NaN, ±inf, negative, huge) plus a hypothesis property.
- Scheduler: low-battery worker never gets work and is logged once; charging bypass; exhausted
  budget sends work to the local worker and is logged; a careless policy's unaffordable chunk is
  refused by the scheduler; dispatcher commits/settles; PONGs feed measured energy; report fields.

**Found and fixed while testing**
- A policy that silently skipped an unaffordable worker left no trace: the scheduler's gate only
  counted compute energy. It now predicts compute + transfer like the policies and logs
  `energy_budget_exhausted` whether the gate or the per-assignment check refuses.
- Full suite 1249/1249 on Python 3.13 (gate) and 3.10.22.

## Phase 9 — Smart scheduling policies (2026-10-15)

**Built**
- `policies/most_reliable.py`: max `R(chunk, d)`, ties by predicted time; energy-aware.
- `policies/tesar.py`: `α · speed_cost + (1 − α) · κ · (−ln R)` per tile, normalized by the local
  time; replication to a reliability target with `max_replicas`; docstring explains `−ln R`.
- `policies/milp_offline.py`: the offline MILP (PuLP, HiGHS/CBC) on a run's `snapshot.json`, the
  static greedy on the same predictions, and a CLI printing both objectives and the gap.
- Registry: `tesar` (default) and `most_reliable` available; scheduler passes the local worker's
  speed to the context and never runs two copies of a task on one device.
- `snapshot.json` gains `rep`, `lambda`, scheduler and energy settings; `report.json` the remaining
  policy parameters.
- `docs/TESAR_ALIGNMENT.md` (with blanks for the paper's equations).
- `highspy` added for pulp 4 (DECISIONS.md); per-round memoization of chunk geometry (perf).
- Gate: `PHASE_CHECKS[9]` (hetero_faulty6 with `tesar`, replicas used, MILP on the real snapshot);
  perf smoke step active from this phase.

**Tested**
- `test_policies.py`: most_reliable picks the reliable device and breaks ties by time; tesar at
  α = 1 / 0 matches fastest / most_reliable and picks the middle device at α = 0.5; cost formula;
  energy-infeasible and low-battery devices never chosen by any policy; local exemption;
  replication triggers below target, stops at `max_replicas` and at the target, never on one
  device, not when disabled or already reliable or local; SUSPECT/QUARANTINED workers never used;
  MILP ≤ greedy for five α values; MILP respects budgets; hostile snapshots rejected; CLI.
- `test_perf_smoke.py` (`perf`): decision for 20 workers × 200 tasks < 20 ms for four policies;
  10 MB protocol round trip < 200 ms.

**Manual check (gate, hetero_faulty6, `tesar`, 1024×1024)** — output correct; 17 of 33 attempts
were replicas (every remote device starts at rep 0.5 < target 0.9); MILP on the run's snapshot:
greedy objective 2.90 vs optimum 1.21. The MILP keeps almost everything local at job start, which
the alignment doc discusses.

**Found and fixed while testing**
- pulp 4 ships no solver and has a new API; added `highspy`, wrote a pulp 3/4 shim (DECISIONS.md).
- `tesar` decision time 14.6 ms → 7.3 ms (memoized chunk geometry) for the perf budget.
- Full suite 1278/1278 on Python 3.13 (gate) and 3.10.22 (pulp 3.3.2 + CBC/HiGHS there).

## Phase 10 — Result verification and malicious worker detection (2026-10-15, milestone)

**Built**
- `coordinator/verifier.py`: probation + reputation-weighted sampling (seeded rng); checks
  re-execute the producing chunk on the trusted local worker (or the best-reputed non-claimant);
  per-tile hash comparison (constant time); strikes with reputation `wrong_result`; trusted
  tiebreak for untrusted disagreement; result swap + taint propagation; quarantine (sticky across
  reconnects) with cancellation and retroactive repair of unverified work; collusion rule for
  matching replicas; free comparison of finished-anyway replicas; skip-instead-of-hang when no
  verifier can exist; statistics.
- Dispatcher: verification/tiebreak results are judged, never stored; failed checks requeued;
  results from quarantined workers ignored (security log).
- Scheduler dispatches checks before policy work; coordinator finishes only when no check is
  outstanding; report: verifications, mismatches, overhead %, detection latency, quarantined,
  repaired tasks, tiebreaks, free replica matches, `wrong_results_reached_final` (vs reference),
  per-worker strikes/verified_ok/state.
- Worker: `--collude-key` lab fault (identical corruption across a colluding group).
- Gate: `PHASE_CHECKS[10]` (hetero_malicious6 → quarantine, correct output).

**Tested**
- `test_verifier.py` (19): probation then sampling; probability formula incl. cap; trusted never
  verified; match; trusted mismatch → strike, reputation, swap, dependents invalidated, security
  event; quarantine after 2 strikes cancels/requeues in-flight work, repairs only its unverified
  results, ignores its later results; untrusted mismatch → trusted tiebreak blames the liar;
  collusion-safe agreement; identical wrong replicas → both struck; reputed replica match = free
  verification; replica disagreement → trusted check; failed check requeued; stale check dropped;
  busy local → wait; unverifiable → skipped; JSON (row stats) results; disabled → no checks.
- `test_e2e_malicious.py` (5, real processes): 100 % liar quarantined + correct output (no wrong
  tile reached final); 20 % liar with `base_prob = 1` → correct; verification off → **incorrect**
  output (negative control); two colluders (`--collude-key`) verify each other and agree on
  identical wrong results — the collusion rule refuses that agreement and a trusted tiebreak
  quarantines both; quarantine survives reconnect and the worker gets nothing in the next job.
- Worker: colluding corruption is identical per input; `--collude-key` requires `--lab-mode`.

**Manual check (gate, hetero_malicious6, tesar, verify all)** — output correct; `bad-phone` (30 %)
quarantined in every run, `sneaky-phone` (5 %) in some; ~16 % verification overhead; detection
latency ≈ 0.4 s after the first wrong result; 20–64 tasks repaired retroactively.

**Found and fixed while testing**
- The audit log's event whitelist rejected the new `wrong_result` event inside the result handler
  (the unit tests had stubbed the log); unit tests now spy on the real log.
- With every remote worker quarantined and `allow_local=false` the job hung: the trusted local
  worker now takes over (DECISIONS.md).
- The negative-control and quarantine e2e tests depended on replica races; they run without
  replicas so the liar's results are certainly stored.
- `test_cancel_queued_and_running` had become a race on the faster CPU (DECISIONS.md).
- Full suite 1305/1305 on Python 3.13 (gate, e2e ×3) and 3.10.22. One earlier 3.10 run had the
  perf smoke tests fail while two browser renderers used ~150 % CPU (load average 9.7); they pass
  in isolation and in the re-run — the 20 ms budget assumes an otherwise idle laptop.

## Phase 11 — Live dashboard (2026-10-16)

**Built**
- `mdc/dashboard/auth.py`: admin token (0600 file, created on first run), constant-time token and
  CSRF checks, server-side sessions with expiry and a size bound, per-address login rate limit,
  Host and Origin checks.
- `mdc/dashboard/headers.py`: CSP without `unsafe-inline`, nosniff, DENY framing, no-referrer,
  COOP, `no-store` on the API; generic JSON errors, never a stack trace.
- `mdc/dashboard/web.py`: routes from the spec (login/logout, state, images, job start with 409,
  demo controls, output image, report, websocket with snapshot + events + 0.5 s updates);
  path-traversal-safe image choice; HTTPS required (coordinator certificate) for a non-loopback
  host, refused at startup otherwise.
- `mdc/dashboard/backend.py`: snapshot (coordinator, workers with rep/λ/RTT/bandwidth/battery/
  energy, grid, per-tile status and producer, events, live metrics, final result), job start,
  CONTROL forwarding.
- Frontend (`index.html`, `app.js`, `style.css`, vanilla, offline): tile canvas coloured by worker
  (running stripes, verifying outline, verified tick, red flash for a replaced wrong tile,
  invalidated outline), worker table with state badges, reputation bars and demo buttons, metrics,
  start-job form, event log with filter, result image with correct/incorrect badge; light/dark,
  responsive.
- Coordinator CLI starts the dashboard unless `--headless` and prints the login URL and token file.
- Gate: `PHASE_CHECKS[11]` (CLI-served dashboard: public healthz, 401 without session, token login,
  strict cookie, CSP).

**Tested**
- `test_dashboard.py`: state structure, images, job start → 202 then 409, bad bodies → 400,
  report/output, websocket snapshot + events + updates, logout, parsers, tile states, and a real
  coordinator running a job started from the dashboard (correct output, report, PNG, controls).
- `security/test_sec_dashboard.py`: 401 everywhere without a session (HTTP and websocket), login
  wrong token / missing or foreign Origin / rate limit, strict cookie (Secure on HTTPS), CSRF
  missing/wrong and bad Origin → 403, bad Host → 400, all security headers and a CSP without
  `unsafe-inline`, path traversal and symlinks → 400, demo controls 403 unless lab mode + switch,
  non-loopback without TLS refuses to start (function and CLI), no details in error bodies (500,
  bad JSON, form body, 413), foreign-origin websocket → 403, no DOM sinks or inline code in the
  frontend, private token file, session expiry/bound/logout, limiter bound.

**Manual check** — `python -m mdc.coordinator` + `python scripts/local_cluster.py
configs/cluster_profiles/hetero_faulty6.yaml`, open the printed URL, paste the token from
`state_dir/admin_token`, start a job; with `--set security.lab_mode=true --set
dashboard.demo_controls=true` and workers started with `--allow-demo-control`, press "Crash" or
"Make malicious" and watch the tiles return to pending or the worker get quarantined.

**Found and fixed while testing** — aiohttp 3.14's string request-key warning (typed key now);
bandit's `urlopen` finding in the new gate check (replaced by `http.client` on loopback, no
suppression). Full suite 1337/1337 on Python 3.13 (gate) and 3.10.22.

## Phase 12 — Android/Termux deployment (2026-10-16)

**Built**
- `scripts/package_worker.sh` / `package_worker.py`: deterministic `dist/mdc-worker.zip` (worker,
  common, tasks, `requirements-worker.txt`, `termux_setup.sh`, `WORKER_README.txt`) plus
  `mdc-worker.zip.sha256`; refuses to package secrets.
- `scripts/termux_setup.sh`: packages, wake lock, battery check, next steps.
- `docs/DEPLOYMENT_TERMUX.md`: F-Droid installs and battery settings; zip over HTTP with hash
  check (zip only); join code by QR (`enroll --qr` + `termux-clipboard-get`) or USB; start,
  `--diag`, revoke; network troubleshooting (firewall, client isolation, blocked beacons, wrong
  interface); measuring energy; the real-device acceptance checklist.
- `python -m mdc.coordinator --diag`: interfaces, advertised address, TCP and dashboard port
  availability, discovery settings, certificate fingerprint, device count, one test beacon; no
  secrets. The worker `--diag` already existed (Phase 3).
- `python -m mdc.cli calibrate-energy --runs DIR`: per-class least-squares suggestions for the
  energy constants; report per worker gains `compute_s` / `transfer_s`.
- Gate: `PHASE_CHECKS[12]` (zip builds, unpacked worker runs `--diag` without coordinator code,
  coordinator `--diag` clean).

**Tested** (`test_deployment.py`, security marker): zip contents (only worker-side code, no
bytecode, no credential files), no secret patterns and a correct `.sha256` line, reproducible
hash, the packager refuses planted secrets, the unpacked zip imports and runs `--diag` with no
coordinator/dashboard/yaml/aiohttp/PIL module loaded, coordinator `--diag` shows no join code
or token, port check, calibration recovers known constants, needs enough data, non-negative fit.

**Real-device acceptance — still to do** (needs physical phones), see
`docs/DEPLOYMENT_TERMUX.md` §8: two phones + local worker complete an image job with correct
output; one phone leaves WiFi mid-job and the job still completes; record in OBSERVATIONS.md.

## Phase 13 — Experiment framework and plots (2026-10-16)

**Built**
- `experiments/run_matrix.py`: validated YAML configs (`yaml.safe_load`), cartesian matrix ×
  repetitions, optional local baseline, `cluster.<flag>` worker options, fresh port/state per run,
  coordinator + `local_cluster.py` with the same seed, resume, one CSV row per run (run identity,
  matrix parameters, §11.3 metrics), `--dry-run`, `--only N`.
- `experiments/plots.py`: mean ± 95 % CI (t distribution) figures as PNG + PDF for E1–E5 and E7
  (MILP solved on saved snapshots), `summary.csv` per point.
- Configs `e1_baselines`, `e2_alpha_sweep`, `e3_failure_sweep`, `e4_chunking`, `e5_malicious`,
  `e7_milp_small` with expected runtime in their headers; `docs/EXPERIMENTS.md` (questions,
  hypotheses, how to run and read, threats to validity).
- `local_cluster.py --extra-args`.
- All six experiments run once in full (145 runs, all ok); figures and a written reading in
  `docs/results/`.
- Gate: `PHASE_CHECKS[13]` (10 real E1 runs → CSV rows, figures, CIs).

**Tested** (`test_experiments.py`): shipped configs parse and stay laptop-sized with documented
runtime; plan (count, ids, seeds, baseline); cluster-key mapping; ten kinds of bad config
rejected; dry run creates nothing; CSV schema and one row per run; CI against the hand-computed
example (2, 4, 6, 8 → 5 ± 4.1085); helpers; every figure type written; E7 MILP on snapshots; a real
two-run experiment with resume and plotting (e2e).

**Findings worth reading** — `docs/results/README.md`: on one laptop `local` is fastest
(simulation limit), `tesar` is the fastest and most predictable offloading policy and degrades
least under drops, chunking and replication interact, and a 5 % liar slipped through once at
`base_prob = 0.05`.

## Phase 14 — Documentation, demo and polish (2026-10-16, final milestone)

**Done**
- README rewritten for a new reader: summary and citation, features, architecture diagram,
  install, the three-command laptop-only quick start, dashboard (screenshot placeholder), real
  phones, enrollment, tests, experiments, repository map, documentation index, credits and the
  AI-assistance section left to fill in.
- `docs/ARCHITECTURE.md` (components, data flow of one task, state machines, concurrency model,
  design decisions), `docs/DEMO_SCRIPT.md` (5-minute script + fallbacks), `docs/TESTING.md`
  (phase checks), `SECURITY.md` (status, measured residual risk, two added residual risks).
- Soak test (`tests/test_soak.py`, 40 chaotic jobs) and the gate's soak invocation fixed.
- `PHASE_CHECKS[14]`: README quick start in a clean checkout → correct output.
- Polish: no placeholders, every module documented, public functions typed (now a test),
  `compileall` clean.

**Concurrency and robustness checklist (spec §12), re-verified at Phase 14**
- [x] No blocking work on the event loop: kernels, image loading, reference, stage 3 and PNG
      encoding run in executors. The dashboard does only small reads on the loop (directory
      listing, token file at start-up).
- [x] One reader task and a write lock per connection.
- [x] Every network read has a timeout or runs in a cancellable task.
- [x] Background tasks are tracked and cancelled; an autouse fixture fails any test that leaves
      a pending task to be destroyed, and the soak test bounds the task count.
- [x] A result is accepted at most once (attempt status + task status + epochs).
- [x] Jobs never hang: timeouts, loss, cancellation, the max-attempts local fallback, and (since
      Phase 10) the local takeover when no remote worker is left; verification checks are
      requeued on failure or skipped when no verifier can exist.
- [x] Unknown message types are ignored and malformed frames close the connection
      (fuzz tests, Phase 1).
- [x] Experiment randomness uses seeded `random.Random` instances (`rng`), never for security.
- [x] Monotonic clocks everywhere except human-readable `wall` timestamps.

**Final verification** — `python scripts/phase_gate.py --phase 14 --repeat-e2e 3`: PASS (e2e ×3,
perf, soak, 1460 tests, 94.8 % coverage, security-critical modules ≥ 97 %, all 15 phase checks).
Full suite 1460/1460 on Python 3.10.22 as well. The gate's soak step had a latent bug, found and
fixed (DECISIONS.md).

**Remaining:** the real-phone acceptance checklist (`docs/DEPLOYMENT_TERMUX.md`
§8), the paper's equations in `docs/TESAR_ALIGNMENT.md`, a dashboard screenshot, and the
AI-assistance disclosure in README.md.

**Future work and research directions**
- **Several competing offloaders** sharing the same volunteer devices: incentives and pricing
  (link to the group's edge-federation Stackelberg work), so devices are paid for energy and
  reliability.
- **Mobility prediction:** anticipate a phone leaving WiFi (signal strength, movement) and drain
  its queue early instead of waiting for DEAD.
- **Learning-based failure prediction** from the event logs (per-device hazard models instead of
  the Beta/λ estimate), and per-direction bandwidth estimation for asymmetric links.
- **Real radio-level energy measurement** (power monitor or Android's battery stats per UID)
  to calibrate `p_tx` properly; include fixed per-offload costs in the speedup model (see
  OBSERVATIONS.md).
- **Transport:** TLS 1.3-only with session resumption for flaky phones; optional QUIC.
- **Verification:** deterministic redundancy for high-value tasks, and a reputation-weighted
  choice of untrusted verifiers when no trusted machine is available.
- **Normalization of the TESAR cost** (online per-task vs offline makespan; TESAR_ALIGNMENT.md §5)
  and comparison with the paper's exact formulation once its equations are filled in.
- **Real-device experiments** (Phase 12 checklist) before drawing performance conclusions.

### Gate run — phase 0 — 2026-10-07T05:56:58+00:00

**PHASE 0 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 22 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 100.0% (threshold 70%)
- 11 security: PASS — security suite green
- 12 e2e: SKIP — e2e tests start at Phase 4
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: SKIP — no security-critical module due yet
- 15 skip-guard: PASS — 234 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600

### Gate run — phase 1 — 2026-10-07T07:18:35+00:00

**PHASE 1 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 35 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 99.4% (threshold 70%)
- 11 security: PASS — security suite green
- 12 e2e: SKIP — e2e tests start at Phase 4
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, tls.py 100%, validation.py 100%, enrollment.py 99%
- 15 skip-guard: PASS — 708 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused

### Gate run — phase 1 — 2026-10-07T12:57:41+00:00

**PHASE 1 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 7 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 99.4% (threshold 70%)
- 11 security: PASS — security suite green
- 12 e2e: SKIP — e2e tests start at Phase 4
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, tls.py 100%, validation.py 100%, enrollment.py 99%
- 15 skip-guard: PASS — 708 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused

### Gate run — phase 2 — 2026-10-07T13:27:15+00:00

**PHASE 2 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 24 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 99.2% (threshold 70%)
- 11 security: PASS — security suite green
- 12 e2e: SKIP — e2e tests start at Phase 4
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, tls.py 100%, validation.py 100%, enrollment.py 99%
- 15 skip-guard: PASS — 913 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks)

### Gate run — phase 3 — 2026-10-07T14:06:57+00:00

**PHASE 3 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 7 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 97.2% (threshold 70%)
- 11 security: PASS — security suite green
- 12 e2e: SKIP — e2e tests start at Phase 4
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, tls.py 100%, validation.py 100%, enrollment.py 99%
- 15 skip-guard: PASS — 1035 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output

### Gate run — phase 4 — 2026-10-07T14:39:18+00:00

**PHASE 4 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 32 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 94.7% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 99%
- 15 skip-guard: PASS — 1117 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0759 s)

### Gate run — phase 5 — 2026-10-07T15:33:50+00:00

**PHASE 5 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 29 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 94.9% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x3
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%
- 15 skip-guard: PASS — 1165 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.2326 s) | P5: crash + drop workers: output_correct, 5 timeouts, 2 lost, makespan 14.1665 s

### Gate run — phase 6 — 2026-10-08T18:57:28+00:00

**PHASE 6 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 32 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.1% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%
- 15 skip-guard: PASS — 1208 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.2284 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 9.0413 s | P6: hetero5/fastest: output_correct, 8 attempts, chunk sizes [1, 4], 6 probes

### Gate run — phase 7 — 2026-10-15T03:10:57+00:00

**PHASE 7 GATE: FAIL (step 10)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 13 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: FAIL — pytest (unit/property/integration) exited with 1

### Gate run — phase 7 — 2026-10-15T03:16:27+00:00

**PHASE 7 GATE: FAIL (step 12)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 14 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.1% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: FAIL — pytest -m e2e (run 1/1) exited with 1

### Gate run — phase 7 — 2026-10-15T03:23:01+00:00

**PHASE 7 GATE: FAIL (step 12)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 15 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.1% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: FAIL — pytest -m e2e (run 1/1) exited with 1

### Gate run — phase 7 — 2026-10-15T03:28:07+00:00

**PHASE 7 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.2% (threshold 80%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%
- 15 skip-guard: PASS — 1228 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0567 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.49 s | P6: hetero5/fastest: output_correct, 8 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.94, dropper-phone 0.42, fast-phone 0.95, flaky-phone 0.85, laptop-b 0.95, local 0.94, slow-phone 0.86

### Gate run — phase 8 — 2026-10-15T03:42:15+00:00

**PHASE 8 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.2% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: SKIP — perf from Phase 9, soak at Phase 14
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%
- 15 skip-guard: PASS — 1249 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.1041 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.491 s | P6: hetero5/fastest: output_correct, 8 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.94, dropper-phone 0.49, fast-phone 0.94, flaky-phone 0.85, laptop-b 0.95, local 0.95, slow-phone 0.86 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.0907 J of 2430.0 J budget, measured None J

### Gate run — phase 9 — 2026-10-15T03:55:57+00:00

**PHASE 9 GATE: FAIL (step 10)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 22 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: FAIL — pytest (unit/property/integration) exited with 1

### Gate run — phase 9 — 2026-10-15T04:05:47+00:00

**PHASE 9 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 22 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.1% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: PASS — perf ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%
- 15 skip-guard: PASS — 1278 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0731 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.7392 s | P6: hetero5/fastest: output_correct, 8 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.94, dropper-phone 0.42, fast-phone 0.95, flaky-phone 0.83, laptop-b 0.95, local 0.94, slow-phone 0.86 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.08 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 34 attempts, 17 replicas, makespan 0.4045 s; MILP greedy gap: 181.48 %

### Gate run — phase 10 — 2026-10-15T04:33:01+00:00

**PHASE 10 GATE: FAIL (step 12)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 23 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.1% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: FAIL — pytest -m e2e (run 2/3) exited with 1

### Gate run — phase 10 — 2026-10-15T04:57:44+00:00

**PHASE 10 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 24 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.0% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x3
- 13 perf/soak: PASS — perf ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%, verifier.py 97%
- 15 skip-guard: PASS — 1305 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0417 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.4824 s | P6: hetero5/fastest: output_correct, 14 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.94, dropper-phone 0.42, fast-phone 0.95, flaky-phone 0.87, laptop-b 0.96, local 0.98, slow-phone 0.86 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.0426 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 37 attempts, 21 replicas, makespan 0.3971 s; MILP greedy gap: 144.27 % | P10: hetero_malicious6/tesar: output_correct, quarantined ['bad-phone', 'sneaky-phone'], 44 verifications (17.336 % overhead), 4 mismatches, 53 tasks repaired, detection [{'worker': 'sneaky-phone', 'latency_s': 0.0892}, {'worker': 'bad-phone', 'latency_s': 0.0912}]

### Gate run — phase 11 — 2026-10-16T03:49:36+00:00

**PHASE 11 GATE: FAIL (step 3)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: FAIL — ruff format exited with 1

### Gate run — phase 11 — 2026-10-16T03:49:43+00:00

**PHASE 11 GATE: FAIL (step 6)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: FAIL — bandit exited with 1

### Gate run — phase 11 — 2026-10-16T03:53:48+00:00

**PHASE 11 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 95.3% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: PASS — perf ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%, verifier.py 97%, auth.py 100%
- 15 skip-guard: PASS — 1337 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0938 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.5931 s | P6: hetero5/fastest: output_correct, 14 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.95, dropper-phone 0.49, fast-phone 0.95, flaky-phone 0.85, laptop-b 0.96, local 0.97, slow-phone 0.88 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.1082 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 37 attempts, 21 replicas, makespan 0.52 s; MILP greedy gap: 183.47 % | P10: hetero_malicious6/tesar: output_correct, quarantined ['bad-phone', 'sneaky-phone'], 43 verifications (16.149 % overhead), 4 mismatches, 73 tasks repaired, detection [{'worker': 'bad-phone', 'latency_s': 0.1193}, {'worker': 'sneaky-phone', 'latency_s': 0.1875}] | P11: dashboard served on http://127.0.0.1:45937: login, session cookie, CSP, 0 worker(s) in state

### Gate run — phase 12 — 2026-10-16T04:07:12+00:00

**PHASE 12 GATE: FAIL (step 7)**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 16 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: FAIL — 1 violation(s)

### Gate run — phase 12 — 2026-10-16T04:11:08+00:00

**PHASE 12 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 16 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 94.8% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: PASS — perf ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%, verifier.py 97%, auth.py 100%
- 15 skip-guard: PASS — 1347 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0752 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.5938 s | P6: hetero5/fastest: output_correct, 15 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.95, dropper-phone 0.49, fast-phone 0.96, flaky-phone 0.85, laptop-b 0.96, local 0.97, slow-phone 0.88 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.1167 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 34 attempts, 18 replicas, makespan 0.5225 s; MILP greedy gap: 78.19 % | P10: hetero_malicious6/tesar: output_correct, quarantined ['bad-phone', 'sneaky-phone'], 43 verifications (18.536 % overhead), 4 mismatches, 67 tasks repaired, detection [{'worker': 'sneaky-phone', 'latency_s': 0.178}, {'worker': 'bad-phone', 'latency_s': 0.1983}] | P11: dashboard served on http://127.0.0.1:51981: login, session cookie, CSP, 0 worker(s) in state | P12: worker zip 36 files, 79 KiB, sha256 aac0672ce06f…; diags ok

### Gate run — phase 13 — 2026-10-16T04:56:38+00:00

**PHASE 13 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 18 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 94.8% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x1
- 13 perf/soak: PASS — perf ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 99%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%, verifier.py 97%, auth.py 100%
- 15 skip-guard: PASS — 1377 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0934 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.3433 s | P6: hetero5/fastest: output_correct, 14 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.96, dropper-phone 0.42, fast-phone 0.95, flaky-phone 0.86, laptop-b 0.96, local 0.97, slow-phone 0.86 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.1272 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 36 attempts, 20 replicas, makespan 0.5185 s; MILP greedy gap: 151.58 % | P10: hetero_malicious6/tesar: output_correct, quarantined ['bad-phone'], 44 verifications (16.167 % overhead), 4 mismatches, 20 tasks repaired, detection [{'worker': 'bad-phone', 'latency_s': 0.2012}] | P11: dashboard served on http://127.0.0.1:44537: login, session cookie, CSP, 0 worker(s) in state | P12: worker zip 36 files, 79 KiB, sha256 aac0672ce06f…; diags ok | P13: e1 x10 runs ok, figures + CIs; mean makespan s: local 0.17, tesar 0.53, fastest 1.81, most_reliable 1.83, round_robin 6.28

### Gate run — phase 14 — 2026-10-16T05:24:13+00:00

**PHASE 14 GATE: PASS**

- 1 environment: PASS — Python 3.13.3; tools ok; git: 14 changed/untracked path(s)
- 2 compile: PASS — mdc, scripts, experiments
- 3 format: PASS — formatted
- 4 lint: PASS — E,F,W,I,B,UP,S,SIM,C4,RUF,ASYNC,PL clean
- 5 types: PASS — mypy clean
- 6 sast: PASS — no medium/high findings
- 7 forbidden: PASS — clean; 0 reviewed exemption(s)
- 8 isolation: PASS — worker side = stdlib + numpy
- 9 dep-audit: PASS — no known vulnerabilities
- 10 unit+prop+integ: PASS — coverage 94.8% (threshold 85%)
- 11 security: PASS — security suite green
- 12 e2e: PASS — e2e green x3
- 13 perf/soak: PASS — perf ok, soak ok
- 14 critical-cov: PASS — auth.py 100%, credentials.py 100%, protocol.py 100%, ratelimit.py 100%, tls.py 100%, validation.py 100%, enrollment.py 100%, verifier.py 97%, auth.py 100%
- 15 skip-guard: PASS — 1460 tests, 0 skipped (all known)
- 16 phase-checks: PASS — P0: config ok; 4 images; state dir 0700 / files 0600 | P1: cert ok; enroll ok; join code round trip; pin ok / wrong pin refused | P2: reference 1000x1300 == tiled (100 tasks) | P3: worker --diag ok, no secret in output | P4: 512x512 job, 2 workers + local: output_correct (makespan 0.0524 s) | P5: crash + drop workers: output_correct, 5 timeouts, 0 lost, makespan 8.587 s | P6: hetero5/fastest: output_correct, 14 attempts, chunk sizes [1, 4], 6 probes | P7: hetero_faulty6/round_robin: output_correct, rep_final crasher-phone 0.95, dropper-phone 0.42, fast-phone 0.95, flaky-phone 0.86, laptop-b 0.96, local 0.97, slow-phone 0.88 | P8: 15 % worker idle + worker_ineligible_battery logged; output_correct; gate-w1 0.1351 J of 2430.0 J budget, measured None J | P9: hetero_faulty6/tesar: output_correct, 35 attempts, 19 replicas, makespan 0.4758 s; MILP greedy gap: 125.26 % | P10: hetero_malicious6/tesar: output_correct, quarantined ['bad-phone'], 44 verifications (15.529 % overhead), 3 mismatches, 28 tasks repaired, detection [{'worker': 'bad-phone', 'latency_s': 0.2152}] | P11: dashboard served on http://127.0.0.1:53997: login, session cookie, CSP, 0 worker(s) in state | P12: worker zip 36 files, 80 KiB, sha256 1f1d6ed7199b…; diags ok | P13: e1 x10 runs ok, figures + CIs; mean makespan s: most_reliable 0.27, local 0.28, tesar 0.47, fastest 1.89, round_robin 4.83 | P14: README quick start in a clean checkout (295 files): output correct, 5 workers, policy tesar, makespan 1.0564 s
