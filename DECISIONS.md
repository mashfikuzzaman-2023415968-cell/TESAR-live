# Design decisions and deviations from the spec

Format: date — decision — reason — alternatives considered. Every deviation from
`TESAR_LIVE_BUILD_SPEC.md` and every security-tool suppression is recorded here.

## 2026-10-07 — Repository lives in `tesar-live/` next to the spec
- **Decision:** the repository root is `network_project/tesar-live/` (spec Section 3 tree), with the
  spec file one level up.
- **Reason:** keeps the spec out of the repository's scans and history.

## 2026-10-07 — Extra module `mdc/common/statedir.py`
- **Decision:** the "secure state-dir helper" required by Phase 0 lives in its own module
  (create 0700 dir, atomic 0600 secret files, refuse group/world-readable secrets).
- **Reason:** Section 3 does not name a file for it; it is used by both coordinator
  (`~/.mdc`) and worker (join-file check), so it belongs in `mdc/common`.
- **Alternatives:** putting it in `config.py` (would drag PyYAML into worker code) or
  `credentials.py` (that is join-code specific).

## 2026-10-07 — Extra test file `tests/security/test_sec_config.py`
- **Decision:** the Phase-0 security tests (insecure config combinations, state-dir and secret-file
  permissions) get their own security test file.
- **Reason:** Section 3 lists no file for these Phase-0 controls; every test under
  `tests/security/` carries the `security` marker, as G.1 requires.

## 2026-10-07 — `mdc/common/config.py` is coordinator-only (imports PyYAML)
- **Decision:** `check_worker_isolation.py` has one documented exception allowing `yaml` in
  `mdc/common/config.py`. Workers never import that module; `tests/test_common.py` checks in a
  fresh interpreter that the worker-side common modules do not load `yaml`, `PIL`, `aiohttp` or
  `cryptography`.
- **Reason:** Section 3 places `config.py` in `mdc/common`, while S.11/rule 0.1.5 require worker
  code to be stdlib + numpy. Both are honoured by keeping the module off the worker's import graph.
- **Alternatives:** move config to `mdc/coordinator/` (deviates from the layout), or write a
  YAML-free config parser (needless risk).

## 2026-10-07 — Lock files generated with `uv pip compile --universal --python-version 3.10`
- **Decision:** `requirements.lock` / `requirements-worker.lock` are produced with
  `uv pip compile --universal --python-version 3.10 --generate-hashes` instead of `pip-compile`.
  Regenerate with:
  `uv pip compile --universal --python-version 3.10 --generate-hashes requirements.txt -o requirements.lock`
  (and the same for `requirements-worker.txt`).
- **Reason:** the development machine runs Python 3.13. `pip-compile` resolves only for the running
  interpreter, so its lock would omit dependencies needed on 3.10 (e.g. `async-timeout` for aiohttp)
  and pin versions that dropped 3.10 (numpy ≥ 2.3, scipy ≥ 1.16). CI installs the lock with
  `--require-hashes` on 3.10 and 3.12, which would fail. uv's universal resolution writes
  per-Python-version markers and hashes for every wheel. `pip-tools` stays in the dev requirements.
- **Alternatives:** install Python 3.10 locally just to lock (not available offline); add upper
  bounds by hand (fragile, still misses marker-only dependencies).

## 2026-10-07 — bandit B104 suppression on `network.coordinator_bind` default
- **Decision:** `coordinator_bind: "0.0.0.0"` (spec Section 5) carries `# nosec B104`.
- **Reason:** the coordinator must accept workers from the LAN; binding all interfaces is the
  purpose. Exposure is controlled by TLS, mutual authentication, enrollment, rate limits and
  connection limits (Section S), not by the bind address. The dashboard, which needs no LAN
  access, defaults to `127.0.0.1`.

## 2026-10-07 — ruff `S603` ignored in `scripts/`, `experiments/` and `tests/`
- **Decision:** per-file ignore of S603 ("subprocess call — check for untrusted input").
- **Reason:** these files run our own tools with fixed argument lists (`sys.executable -m ...`),
  never a shell, always with a timeout. S603 fires on *every* subprocess call regardless; the
  dangerous variants (`shell=True`, `os.system`, partial executable paths — S602/S605/S607) remain
  enabled everywhere, and `check_forbidden.py` also bans `shell=True`. Production code under `mdc/`
  keeps S603 enabled, so each of its (few) subprocess calls must be reviewed.

## 2026-10-07 — Gate skip guard reads JSON results, not junit XML
- **Decision:** `tests/conftest.py` writes every test id and skip reason to the file named by
  `MDC_GATE_RESULTS`; `phase_gate.py` reads that JSON.
- **Reason:** avoids parsing XML (bandit B314) without adding the `defusedxml` dependency.

## 2026-10-07 — Config defaults live in the dataclasses; a test pins them to `default.yaml`
- **Decision:** `load_config()` without a file uses the dataclass defaults, and
  `test_defaults_match_default_yaml` asserts they equal `configs/default.yaml` exactly (values
  and key set).
- **Reason:** works no matter where the package is installed, while guaranteeing the YAML file
  (the human-readable reference) never drifts.

## 2026-10-07 — Additional config range checks
- **Decision:** besides types and unknown keys, `config.py` rejects meaningless values: e.g.
  `alpha ∉ [0,1]`, `suspect_after_s ≥ dead_after_s`, `min_chunk_tiles > max_chunk_tiles`,
  and job parameters above `security.max_task_params` (`job.blur_radius ≤ max_task_params.blur_radius`, etc.).
  `workers.min_battery_pct` is typed as a float (YAML `20` is accepted and widened).
- **Reason:** fail closed (rule 0.1.12); a job that the workers would reject anyway should fail at
  startup with a clear message.

## 2026-10-07 — `test_1000x1300.png` is 1000 wide × 1300 high
- **Reason:** "W×H" is the usual image naming convention; either way neither side is a multiple
  of the 128-pixel tile, which is the purpose of the image.

## 2026-10-07 — Phase gate runs the phase checks of *all* phases ≤ N
- **Decision:** `PHASE_CHECKS[k]` for every k ≤ N, not just N.
- **Reason:** turns each earlier phase's acceptance check into a regression test. A missing entry
  for a completed phase fails the gate.

## 2026-10-07 — Forbidden-pattern scanner details
- Python rules use the AST, so `asyncio.create_subprocess_exec(` is not mistaken for `exec(`.
- The "`==`/`!=` on secret-like names" rule matches underscore-separated *name components*
  (`mac`, `digest`, `token`, `secret`, `proof`, `fingerprint`): `device_secret == x` is flagged,
  `tokens == 0` (a rate-limit bucket) is not.
- The chmod rule flags any world-writable mode (a superset of the spec's `0o777|0o666`).
- The repository secret scan skips `.git`, `.venv`, `runs/`, `dist/` and `tests/fixtures/`
  (fixtures are generated at test time and git-ignored).

## 2026-10-07 — Development interpreter is Python 3.13; 3.10 compatibility enforced by tools
- **Decision:** ruff `target-version = "py310"` and mypy `python_version = "3.10"` reject newer
  syntax/stdlib APIs; CI runs the gate on 3.10 and 3.12. The gate's explicit Python ≥ 3.10 check
  is replaced by `requires-python = ">=3.10"` in `pyproject.toml` (ruff UP036 flags the
  version check as dead code under the 3.10 target).

## 2026-10-07 — Phase 1: error class hierarchy
- **Decision:** `WireError` is the base of `ProtocolError`, `IntegrityError` and `ConnectionClosed`;
  `ValidationError` subclasses `ProtocolError`, `UnknownMessageType` subclasses `ValidationError`.
  All live in `common/protocol.py`; `common/validation.py` re-exports `ValidationError`.
- **Reason:** a schema violation is a protocol violation (both justify closing the connection),
  while an integrity failure is corruption in transit and must *not* be handled like a hostile
  peer (spec Phase 4: reschedule the attempt). Defining them in `protocol.py` avoids an import
  cycle (`unpack_array` raises `ValidationError` and `validation.py` needs the protocol constants).

## 2026-10-07 — Phase 1: strict JSON beyond the spec's list
- Duplicate JSON keys and integers longer than 20 digits are rejected in addition to NaN/Infinity,
  deep nesting and oversize strings; `1e999` (which Python parses to `inf`) is rejected too.
- **Reason:** duplicate keys are ambiguous (different parsers keep different values); a
  16 KB integer would hit Python's int-conversion limit and raise an unexpected `ValueError`.

## 2026-10-07 — Phase 1: task parameter names fixed now
- **Decision:** `img_tile` params = `{r, iters}`, `row_stats` = `{edge_threshold}`,
  `synthetic` = `{seed, chain_len}`, exactly these keys.
- **Reason:** S.5.5 requires validating params in Phase 1, before the kernels exist (Phase 2).
  The names follow spec 8.1/8.3/8.5.

## 2026-10-07 — Phase 1: redaction matches name components, not substrings
- **Decision:** a dict key is secret if a component (split on `_ - .`) is `secret`, `token`,
  `password`, `passphrase` or `proof`, if the key is exactly `auth`, or if its last component is
  `key`. Values containing `MDC1.` join codes or PEM private keys are masked everywhere.
- **Reason:** the spec's "keys named like `secret|token|auth|proof|password|key`" taken as
  substrings would mask useful non-secret settings such as `auth_fail_limit`, `key_path` and
  `keyboard`. Over-redaction is safe but makes config dumps useless for reproducing experiments.

## 2026-10-07 — Phase 1: log records are redacted at creation (LogRecordFactory)
- **Decision:** besides a `RedactingFilter` on our handlers, `setup_logging` installs a global
  `LogRecordFactory` that redacts every record when it is created.
- **Reason:** a test showed that pytest's own log-capture handler (any handler we did not
  install) received the raw join code. The factory closes that gap for all handlers.

## 2026-10-07 — Phase 1: forbidden-pattern scanner ignores UPPER_CASE constants
- **Decision:** the "`==`/`!=` on secret-like names" rule skips all-upper-case identifiers.
- **Reason:** `len(secret) != SECRET_BYTES` compares a *length* with a public constant; flagging
  it would force meaningless exemptions. Comparisons with secret *values* are still caught.

## 2026-10-07 — Phase 1: extra security-log event types
- `device_secret_rotated` and `certificate_rotated` are added to the S.10.4 list, so that
  every credential change leaves an audit record.

## 2026-10-07 — Phase 1: `detect_lan_ip` implemented early in `coordinator/beacon.py`
- **Reason:** `mdc.cli enroll` must put the coordinator's address into join codes; the rest
  of `beacon.py` (periodic beacons) is Phase 4.

## 2026-10-07 — Phase 1: `mdc.cli enroll` writes `NAME.join` by default; never prints the code
- Without `--out`, the join file is written to `./NAME.join` (0600). `--qr` prints only the QR
  code. The code text itself is never echoed to the terminal (scroll-back/screen sharing).

## 2026-10-07 — Phase 1: fuzz example counts
- Fuzz tests use 250 Hypothesis examples each (150 for the async stream fuzzer), keeping the
  slowest under ~16 s so they stay well inside the 60 s per-test timeout on slower CI machines.
  Hypothesis's example database makes repeated runs explore further.

## 2026-10-07 — Python 3.10 verified for real; lesson for the Phase-4 server
- **Check:** a Python 3.10.22 interpreter (fetched with `uv`) installed `requirements.lock` with
  `--require-hashes` and ran the full suite: 708/708 pass, three runs in a row.
- **Found:** on 3.10, `asyncio.Server.wait_closed()` does not wait for connection handlers (it does
  from 3.12). A test server whose handler was cancelled leaked its accepted socket — invisible on
  3.13. The test handlers now close their writers in `finally`.
- **Consequence for Phase 4:** the coordinator's server must track every connection and close it
  explicitly on shutdown, not rely on `wait_closed()` (spec Section 12: "all background tasks are
  tracked and cancelled on shutdown").
- Memory-bound tests now use `tests/helpers/memory.peak_extra_bytes`, which measures against a
  reset baseline and is correct even when tracing is already on (`PYTHONTRACEMALLOC`).

## 2026-10-07 — Phase 2: extra module `mdc/tasks/errors.py`
- **Decision:** `TaskError`, `TaskTimeLimitError` and `check_deadline` live in their own module.
- **Reason:** both the kernels and the registry (`base.py`) need them; `base.py` imports the
  kernels, so defining the errors in `base.py` would create an import cycle.

## 2026-10-07 — Phase 2: kernels enforce limits and a cooperative deadline
- Every kernel (`box_blur_valid`, `pipeline_valid`, `reference`, `hash_chain`) checks its parameters
  against `TaskLimits` itself, with exact `int` type checks (bools/floats/numpy scalars rejected),
  so a direct call is as safe as a call through the registry (S.12 row 2).
- Long kernels check an optional `deadline` between blur iterations and every 10 000 hashes,
  raising `TaskTimeLimitError` (`error: "time_limit"`, S.6.5). The worker supplies the deadline in
  Phase 3 (`task_max_s`).

## 2026-10-07 — Phase 2: synthetic task details the spec leaves open
- "seed bytes" = the seed as 8 big-endian bytes (`seed` in 0..2^63-1).
- Seeds of a synthetic job: `experiment.seed * 1_000_003 + i` (distinct per task and per
  experiment seed). Synthetic tasks are never chunked.

## 2026-10-07 — Phase 2: TaskNode/Attempt extra fields and invalidation semantics
- `TaskNode` additionally has `row`, `col`, `params` and `epoch`; `Attempt` has `task_epochs`.
  The spec's field names are all kept.
- `invalidate(t)`: `t` becomes INVALIDATED (runnable again); every transitive dependent that had a
  result or was running loses it and becomes PENDING (so `final` is re-run). Every affected task's
  `epoch` is bumped, and `attempt_is_current()` rejects results of attempts sent before the bump —
  a stage-2 result computed from a since-corrected tile can never be accepted.
- `replace_result(t, ...)` (for Phase 10) keeps `t` DONE with the corrected result and invalidates
  only its dependents.
- VERIFYING counts as "dependency satisfied" (optimistic execution, spec Phase 10 item 3).

## 2026-10-07 — Phase 2: image loading lives in `coordinator/job_builder.py`
- `load_image`/`save_image` (Pillow) are coordinator-only; `mdc/tasks` stays numpy-only for
  phones. `load_image` applies S.9.8 now (Pillow `MAX_IMAGE_PIXELS = 50_000_000`,
  `DecompressionBombError` caught, > 8192 x 8192 refused), not only in the Phase 11 dashboard.

## 2026-10-07 — Phase 2: `execute_inprocess` (in-process job runner)
- Runs a whole job through the same Attempt/registry path as a distributed run, without sockets.
  Used by tests, the P2 gate check and `mdc.cli`; later useful for the local baseline.

## 2026-10-07 — Phase 2: extra test files
- `tests/test_tasks.py` (registry, row_stats, cost model, benchmark), `tests/test_cli.py`
  (reference/compare/bench) and `tests/security/test_sec_kernels.py` (hostile tasks, worker-side
  limits, deadlines). Section 3 has no file for these Phase-2 tests.
- `test_image_pipeline.py` pins the SHA-256 of a reference output (recorded on x86-64): it must be
  reproduced on every platform, including ARM phones (Phase 12 check).

## 2026-10-07 — Phase 3: `--join-code` only accepts `-` (stdin)
- **Conflict in the spec:** the CLI line lists `--join-code CODE`, while the same section and S.10.6
  say secrets must never be passed as an argument (argv is visible in the process list).
- **Decision:** security wins. `--join-code -` reads the code from standard input; any other value
  is refused with an explanation. `--join-file` and `MDC_JOIN_CODE` remain.

## 2026-10-07 — Phase 3: target resolution order
- `--coordinator` → last address that authenticated → UDP beacon (10 s, fingerprint-filtered) →
  the join code's own host:port, after printing the "No beacon heard ..." guidance.
- **Reason:** the join code already contains the coordinator address; using it as the last resort
  keeps workers working on networks that block broadcast, without weakening anything (the pin
  still authenticates the peer).

## 2026-10-07 — Phase 3: worker reactions
- REJECT `auth`/`revoked`/`version`/`protocol` → exit with status 3 (retrying cannot help);
  REJECT `busy` → retry with backoff.
- Pin mismatch or wrong WELCOME proof → logged (`pin_mismatch`, `coordinator_proof_failed`) and
  retried with backoff (the worker never talks to that peer, but a later coordinator may be fine).
- A message failing its schema after authentication → `validation_failed` logged, connection
  closed. A TASK_ASSIGN that is well-formed but exceeds the worker's limits → `TASK_RESULT` error
  (the coordinator learns why). Error strings: `busy`, `battery_low`, `integrity`,
  `invalid_task: ...`, `time_limit`, `internal_error`, or the validation message.
- Worker task limits = element-wise minimum of its own limits and WELCOME `limits` (a coordinator
  can tighten, never loosen). Session slots = min(own slots, `accepted_slots`).
- `compute_ms` includes the `--slowdown` sleep, so an emulated slow device also *looks* slow.

## 2026-10-07 — Phase 3: worker security events are log lines
- The worker writes `SECURITY <event> ...` lines (logger `mdc.worker.security`) to its own rotating
  log; it has no `security.jsonl` (that file belongs to the coordinator's state directory, and
  workers write nothing else to disk, S.7.4). Default log directory: `~/.mdc-worker/logs` (0700,
  files 0600).

## 2026-10-07 — Phase 3: stopping aborts the TLS connection
- Found by a test: `writer.close()` on a TLS stream sends `close_notify` and waits for the peer's
  reply (asyncio default 30 s); the reader stays blocked meanwhile, so SIGTERM took > 15 s.
  `stop()` and the flaky-disconnect fault now use `transport.abort()`; ordinary session ends still
  close gracefully.

## 2026-10-07 — Phase 3: `--diag` implemented now (spec places it in Phase 12)
- The Phase-3 gate check needs `--diag` ("runs and prints no secret"). It prints versions,
  platform, identity (worker id, fingerprint — no secret), battery, benchmark, a beacon listen and a
  TLS + pin connectivity test. Phase 12 may extend it.

## 2026-10-07 — Phase 3: coverage now includes `__main__.py` modules
- The Phase-0 `omit = ["mdc/*/__main__.py"]` dated from one-line placeholders; the worker CLI now
  contains safety logic, so excluding it would hide untested code. The omit was removed.

## 2026-10-07 — Phase 3: Termux battery parsing
- `plugged: "UNPLUGGED"` means not charging even if `status` is unknown (found by a test); the
  battery floor's charging exemption must not be triggered by an unknown status.

## 2026-10-07 — Phase 4: `job_started` uses `job_type`, not `type` (spec conflict)
- The event catalogue (Section 10) gives `job_started` a field `type`, but `type` is already the key
  that names the event in every JSONL line (`{"t", "wall", "type": "job_started", ...}`). Writing
  both is impossible; mypy caught the clash. The field is `job_type` (`"image"`/`"synthetic"`).

## 2026-10-07 — Phase 4: default policy `tesar` refuses to start until Phase 9
- `configs/default.yaml` says `scheduler.policy: tesar`, which is implemented in Phase 9. Until then
  the coordinator exits with status 2 and a message naming the available policies, instead of
  silently substituting another policy (which would invalidate experiments). Phase-4 runs pass
  `--set scheduler.policy=round_robin` (or `local`).

## 2026-10-07 — Phase 4: connection limits and bans apply right after the TLS handshake
- `asyncio.start_server(ssl=...)` performs the TLS handshake before the handler sees the
  connection; the per-IP ban, `max_connections` and `max_connections_per_ip` checks therefore run
  immediately *after* TLS. The TLS handshake itself is bounded by `ssl_handshake_timeout =
  handshake_timeout_s`, so a slowloris peer is still dropped (tested). Rejecting before TLS would
  need manual `start_tls` upgrades (`StreamWriter.start_tls` exists only from Python 3.11).

## 2026-10-07 — Phase 4: what counts towards an IP ban
- Every failed handshake counts: bad MAC, unknown or revoked `worker_id`, invalid/absent HELLO,
  trusted-local spoof, handshake timeout. A successful authentication clears the IP's history.

## 2026-10-07 — Phase 4: scope pulled forward from Phase 5/6 (minimal)
- A PING loop (keepalive; records `last_pong_t` and battery) — without it idle workers drop after
  30 s. SUSPECT/DEAD timing and per-attempt timeouts remain Phase 5.
- TCP close → worker DEAD, its in-flight attempts `lost` and requeued (spec 7.1 "any --TCP closed-->
  DEAD"); needed for any robustness test. Late-result acceptance and max-attempt fallback: Phase 5.
- PROBING → ACTIVE immediately after WELCOME; the bandwidth probe is Phase 6.

## 2026-10-07 — Phase 4: subnet broadcast assumes /24
- Beacons go to `255.255.255.255`, the advertised IP's /24 broadcast address and the multicast group
  (TTL 1). Reading the real netmask portably needs platform-specific code (netifaces / ioctl);
  `network.advertise_host` and `--coordinator` cover networks where /24 is wrong.

## 2026-10-07 — Phase 4: server details
- An unknown `worker_id` triggers one registry re-read before rejecting, so a device enrolled after
  the coordinator started can join immediately (not only after the 2 s poll).
- The worker's display name is taken from the enrollment registry (operator-chosen), not from the
  self-reported HELLO `name`.
- A QUARANTINED worker that reconnects stays QUARANTINED (connection accepted, never scheduled).
- The local worker logs to `state_dir/logs/local.log`; its argv contains no secret (join code via
  `MDC_JOIN_CODE`), and the child gets the package root on `PYTHONPATH`.
- Reference images are cached in `runs/.cache/` (relative to the working directory; the
  `Coordinator(cache_dir=...)` parameter overrides it).

## 2026-10-07 — Phase 4: security-log fields may not be named `type`/`wall`/`suppressed_before`
- Found by a test: `emit("validation_failed", type=msg.type)` overwrote the record's event type
  with a **peer-controlled** string — an attacker could choose which event name appears in the
  audit log. `SecurityLog.emit` now refuses reserved field names (like `EventLogger.emit`), and
  call sites use `msg_type`.

## 2026-10-07 — Phase 4: report naming
- `per_worker[name].tasks_ok` = accepted task results (tiles and row statistics);
  `attempts_ok` = accepted attempts (a chunk is one attempt).

## 2026-10-07 — Phase 4: coordinator exit status
- 0: job finished with correct output (or served until Ctrl+C); 1: job incorrect, failed, timed
  out or interrupted; 2: configuration error (including an unavailable policy).

## 2026-10-07 — Phase 5: report counts attempts by final state
- An attempt can time out and later have its late result accepted; counting status changes as they
  happen double-counts it. `report.json` now counts from each attempt's final status, with an
  `Attempt.timed_out` flag kept for history: `attempts_ok` = results used (`ok` + `late_accepted`),
  `attempts_timeout` = attempts that ever timed out, `attempts_late` = results that arrived after a
  timeout/loss (`late` + `late_accepted`); `attempts_cancelled` added.

## 2026-10-07 — Phase 5: late results
- Accepted only for attempts that timed out or were lost, only if `status: ok`, only from the session
  they were sent to (so in practice: timed-out attempts on a still-connected worker), and only if one
  of their tasks still lacks a result. Late error reports and late invalid results change nothing.

## 2026-10-07 — Phase 5: trusted fallback details
- "Attempts" counts primary and replica attempts (verification attempts excluded). A task with
  `max_attempts_per_task` attempts is reserved for the local worker (never chunked with other tiles,
  never offered to the policy) and `task_fallback_local` is emitted once per task.
- Without a local worker (`workers.local_worker: false`) there is nothing trustworthy to fall back to:
  a warning is logged once and the task stays with the policy.

## 2026-10-07 — Phase 5: error backoff (found by a real failure)
- A worker returning error results (e.g. `battery_low`, `busy`) was re-sent tasks instantly — a hot
  loop of ~10 000 attempts per minute. Error results now put the worker into an exponential cooldown
  (0.25 s, 0.5 s, ... capped at 10 s) during which it is not eligible; a success resets it.

## 2026-10-07 — Phase 5: the trusted local worker has no battery floor
- It is the laptop running the coordinator; spec Phase 8 exempts it from energy limits. It is started
  with `--min-battery 0`. (Found when the development laptop's battery read 2 % on AC power.)

## 2026-10-07 — Phase 5: sysfs "charging" means externally powered
- An online `Mains` power supply counts as charging even when the battery says "Not charging"
  (charge thresholds/battery care). Matches Termux's use of `plugged`. See OBSERVATIONS.md.

## 2026-10-07 — Phase 5: local clusters and the per-IP connection limit
- Every worker of a one-machine cluster connects from 127.0.0.1, so `network.max_connections_per_ip`
  (default 4, a LAN-appropriate value) must be raised for clusters of more than 3 workers. The limit
  is kept; the coordinator now also logs a warning naming the setting, and `local_cluster.py` prints
  the value to use.

## 2026-10-07 — Phase 5: local_cluster.py details
- Re-running a profile re-keys already enrolled names (`rotate_secret`: same `worker_id`, so
  reputation history is kept) instead of failing on "already enrolled".
- Workers default to `--battery mock` unless the profile chooses a provider (a simulated cluster must
  not read the laptop's real battery). `--duration S` stops the cluster after S seconds.
- `chaos.py --events FILE` starts its clock at the coordinator's `job_started` event.

## 2026-10-07 — Phase 5: RTT samples from PONGs are sanity-checked
- `t_c` is echoed by the worker; an RTT that is negative or above 600 s is discarded instead of used
  (a peer could otherwise feed absurd values into Phase 6 estimates).

## 2026-10-08 — Phase 6: `fastest` pulled forward from Phase 9 (spec tension)
- Phase 6 says round-robin and local keep fixed chunking, yet its acceptance requires chunk sizes to
  "visibly differ per worker" — impossible without a policy that uses adaptive chunking, and none
  exists before Phase 9. `fastest` (minimum predicted time, adaptive chunks) is the natural consumer
  of the Phase-6 predictions, so it is implemented now; Phase 8 adds its energy-feasibility check.
- It compares workers by **predicted time per tile** (`predicted_s / n_tiles`): each worker is
  offered its own adaptive chunk size, and comparing whole-chunk times would always favour the worker
  offered the smallest chunk. TESAR's cost (Phase 9) normalizes the same way (`Ts / n`).

## 2026-10-08 — Phase 6: timing never trusts the worker's echo
- The spec computes RTT as `now - t_c` and the probe time from the echoed `t_c`. A dishonest worker
  could fake `t_c` to look faster and attract work. The coordinator instead remembers when it sent
  each PING (`seq`, bounded history of 64) and each PROBE, and times them itself; the echoed `t_c`
  must match exactly or the sample is discarded (it still counts as a liveness PONG).
- The handshake (CHALLENGE sent → HELLO received) provides the first RTT sample.

## 2026-10-08 — Phase 6: probe details
- The probe (256 KiB of `os.urandom` bytes — incompressible) is sent in a background task after the
  handshake, so it does not count against `handshake_timeout_s` and does not stop the reader loop.
  The worker is PROBING (not eligible) until the PROBE_ACK with the right `probe_id` and
  `received_bytes`, or `PROBE_TIMEOUT_S = 3 s` (then ACTIVE with the default bandwidth;
  `probe_done.timed_out = true`).

## 2026-10-08 — Phase 6: predictions everywhere
- The dispatcher now predicts every attempt as `queue_wait + rtt + wire_bytes / bw + compute`, also for
  `round_robin`/`local`/fallback assignments, so timeouts (`factor × predicted + slack`) account for
  queueing and transfer. Chunk wire bytes use the exact chunk geometry (one shared halo) times the
  observed per-kind, per-direction compression ratio.
- `attempt_sent` additionally records `est_rtt_s` and `est_bw_bytes_per_s` (the estimates used).

## 2026-10-08 — Phase 6: limitations of the bandwidth estimate (measured, documented)
- The spec's passive formula divides `bytes_in + bytes_out` by a transfer time; with app-level
  shaping only the worker's *sends* are slowed, so input bytes are counted against output time and a
  1.5 Mbit/s shaped link was estimated at ~3 MB/s — lower than an unshaped link (the ordering is
  right) but far above the truth. Real radio links are slow in both directions.
- In-process tests (coordinator and workers on one event loop) contaminate timings; chunk-size
  behaviour is therefore asserted with real worker processes (`test_e2e_chunking.py`).
- On this development laptop a tile computes in ~2 ms, so with the default `target_task_s = 2 s`
  every worker earns the maximum chunk; tests and the P6 gate check use `target_task_s = 0.1` to put
  the rule in its operating range. Real phones (Phase 12) are where the default matters.

## 2026-10-08 — Phase 6: report and snapshot
- `per_worker.avg_rtt_ms` and `bw_kbps` now come from the estimator; `snapshot.json` includes each
  worker's network estimates and the compression table.

## 2026-10-08 — Phase 6: SIGSTOP e2e test corrected (G.6 rule 2)
- `test_frozen_worker_sigstop_sigcont` asserted `attempts_lost >= 1`, i.e. that the frozen worker's
  work is recovered through DEAD detection. With Phase-6 predictions the attempt timeout
  (`3 × predicted + 0.5 s` ≈ 1.9 s here) fires before `dead_after_s` (2 s), so the same work comes back
  as `timeout` instead. Both are correct recoveries; which wins is a timing race the test must not
  depend on. It now asserts `attempts_lost + attempts_timeout >= 1` and still requires the worker to
  go SUSPECT → DEAD and reconnect.

## 2026-10-09 — Phase 6: cancellation-safe waits (Python 3.10 shutdown hang)
- **Found:** on Python 3.10 two tests hung in teardown for minutes. Diagnosis (watchdog thread dumping
  stacks and asyncio tasks): `Coordinator.stop()` awaited the cancelled `Scheduler.run()` task, which
  kept running. Its loop waited with `asyncio.wait_for(event.wait(), 0.5)`; on Python <= 3.11
  `wait_for` swallows a cancellation that arrives right after its inner wait completed (CPython
  gh-86296, fixed by the 3.12 rewrite). A worker disconnecting during shutdown sets the scheduler's
  wake event just before the cancel — exactly that case. Production impact: a coordinator on 3.10
  could hang on shutdown.
- **Fix:** `mdc.common.timeutil.wait_event(event, timeout)` built on `asyncio.wait` (never hides the
  caller's cancellation), used by the scheduler loop, `run_job` and the worker's reconnect sleep. The
  scheduler also stops via a flag + wake-up, not cancellation alone. Server/worker reader loops were
  already safe (they also end when the connection closes); one-shot `wait_for`s are harmless.
- **Regression test** (`test_scheduler_start_stop_never_hangs`): trigger then stop, 300 times.
  Verified to *fail* (hang) with the old code on 3.10, pass with the old code on 3.13, and pass with
  the fix on both.

## 2026-10-15 — Phase 7: form of the failure rate λ
- The spec suggests `λ = failures_ewma / max(1, busy_time_ewma)` and leaves the exact form open.
  Our tasks last 0.05–2 s, so `max(1, busy)` would make λ ≈ "failures per attempt", not per second,
  and `exp(-λ·T)` would barely depend on `T`. Instead, per finished attempt we keep EWMAs of
  `p` (did it fail: 0/1) and `T` (its predicted duration) and invert the Poisson model the reliability
  formula already assumes: `P(fail in T) = 1 - exp(-λT) = p`, so `λ = -ln(1 - p) / T` (`p ≤ 0.999`,
  `T ≥ 50 ms`, `λ ≥ 1e-6`). Default `default_failure_rate_per_s` until 3 attempts, as specified.
- An attempt counts once: a late result accepted after a timeout adds a reputation success (it was
  valid work) but not a second attempt sample (the timeout already counted it as failed).
- Battery eligibility for `R = 0` (`battery_eligible`) treats an unknown level (desktop, no sysfs)
  as eligible and charging as bypassing the floor; Phase 8 adds the energy budget on top.

## 2026-10-15 — Phase 7: plumbing
- `build_report` was at ruff's argument limit (PLR0913/PLR0917); the optional estimators now travel
  in a `RunModels` dataclass (`netest`, `reputation`; energy and verifier join later) instead of
  suppressing the rule.
- The scheduler takes the reputation model as a constructor argument (like `netest`), shared with
  the dispatcher, so test doubles of the dispatcher keep working.
- `test_reputation_stub` (Phase 5) asserted the stub's exact snapshot dict; it now checks the same
  counts plus `rep` in the larger snapshot (name kept, assertion not weakened).
- The P7 gate check uses `round_robin` with single-tile chunks on a 1024×1024 image so that every
  worker in `hetero_faulty6` builds a history; with `fastest` the slow and faulty devices get almost
  no work and all reputations stay near the 0.5 prior, which shows nothing.

## 2026-10-15 — Phase 7: chunking e2e test calibrated to CPU speed (G.6 rule 2)
- `test_fast_worker_gets_bigger_chunks` began failing — on the unchanged Phase 6 code too (checked
  with `git stash`). Cause: with the laptop now on mains power at full clock speed, a 128 × 128 tile
  with 16 blur iterations costs ~4 ms, far below `target_task_s = 0.1`, so both workers earned the
  maximum chunk before any measurement and the job ended before the slow worker got a second chunk.
  The test passed earlier only because the CPU was throttled.
- Fix: the workload is calibrated from `run_benchmark()` so a 128 × 128 tile costs ~10 ms on the
  fast worker (iterations capped at the security limit 64). The assertions are unchanged; the test
  now measures the chunking rule rather than the CPU governor.
- Emulation fix found on the way: `--slowdown F` slowed computation but the worker still reported
  its real benchmark, so the coordinator's *first* chunks treated an emulated slow phone as fast. A
  real slow phone benchmarks slow, so the reported `bench_ms_per_unit` is now multiplied by `F`
  (lab-mode fault only; test `test_slowdown_fault_slows_the_reported_benchmark`).
- Same cause in `test_crashing_worker` (also failing on the unchanged Phase 6 code): the 16-tile job
  finished before the crash at 2.5 s, so the worker was never declared DEAD. The test now uses 64-tile
  jobs (`job.base_tile=64`) and additionally asserts `attempts_lost + attempts_timeout >= 1`, i.e.
  that the crash really hit in-flight work — a stronger check than before.

## 2026-10-15 — Phase 8: energy model choices
- **Budget** (`capacity · pct · max_job_energy_pct`) is fixed the first time a worker is considered
  during a job (job start for connected workers, first sight for late joiners). A worker that is
  **charging** or reports **no battery level** (desktop) has no budget limit: the job does not drain
  a battery. The trusted local worker is exempt from budget and battery floor (spec item 3).
- **Ledger:** a sent attempt commits its predicted energy; a successful one is then charged the
  energy computed from measured times (worker `compute_ms`, already clamped to the coordinator's
  measured total; the rest of the round trip as transfer). **Failed** attempts (timeout, lost,
  error, cancelled) are charged their prediction: the device most likely did (some of) the work, and
  charging it keeps the budget conservative. A late result after a timeout is not charged twice.
- **Enforcement for every policy:** the scheduler drops remote workers that are below the battery
  floor (not charging) or cannot afford the next ready task (compute + transfer, as policies predict),
  and re-checks each assignment against the live ledger whatever the policy returned. `fastest`
  and `round_robin` also skip unaffordable workers themselves so the work goes elsewhere in the
  same round. `round_robin` therefore no longer "ignores energy entirely": protecting the volunteers'
  batteries is a safety rule, not a scheduling preference.
- Events: `worker_ineligible_battery` (once per job, logged again after the worker recovered and
  dropped below the floor again) and `energy_budget_exhausted` (once per job).
- **Measured energy:** `|current_ma|/1000 · voltage_v · dt` between PONG samples, `dt` on the
  coordinator's clock capped at 3 × `energy.sample_interval_s`; current capped at 20 A and voltage at
  30 V; charging samples skipped. Samples before the job form the idle baseline (`idle_baseline_w`
  in the report), not subtracted.
- **Plumbing:** the Dispatcher and Scheduler hit ruff's argument limit again, so the network,
  reputation and energy models now travel together as `Estimators` (one per run, shared by
  dispatcher, scheduler, failure detector and report); it replaces Phase 7's `RunModels`.
- The constants in `EnergyConfig` are still the spec's rough placeholders; Phase 12's calibration
  helper is where real values come from.

## 2026-10-15 — Phase 9: policies
- **`scheduler.max_replicas` counts extra copies** (replicas), not total copies: with the default 2,
  a chunk runs on at most three devices. Replicas are added in cost order until
  `1 − Π(1 − R_i) ≥ reliability_target`; never on the primary's device (the scheduler also refuses
  two in-flight attempts of one task on one worker), never for a trusted-local primary.
- **`T_loc`** uses the trusted local worker's measured speed (even when it is busy, so the context
  carries it separately); without a local worker, the mean benchmark of the eligible workers stands
  in — only ratios between devices matter.
- **`most_reliable`** uses adaptive chunks like `fastest` and compares `R(chunk, d)`; ties by
  predicted time, then worker id. With fixed one-tile chunks `tesar(α = 0)` makes the same choice
  (the policy tests check this); with adaptive chunks `tesar` compares per tile.
- **Tests updated in place** (names kept): `test_make_policy` now checks that `tesar` (the default)
  and `most_reliable` exist and that an unknown name raises; `test_coordinator_cli_refuses_
  unimplemented_policy` now passes `scheduler.policy=greedy`, which config validation refuses with
  exit code 2 and the list of real policies.
- `report.json` also records `reliability_target`, `max_replicas` and `allow_local`;
  `snapshot.json` records each worker's `rep`/`lambda` and the scheduler and energy settings, so
  the offline MILP sees exactly what the live policy saw.

## 2026-10-15 — Phase 9: MILP solver dependency (`highspy`)
- pulp 4.0 (installed for Python ≥ 3.12 by our universal lock) no longer bundles the CBC solver and
  found no solver at all; pulp 3.3.2 (Python < 3.12) still has CBC. `highspy` (HiGHS, MIT license,
  binary wheels) was added to `requirements.txt` and the lock regenerated with the documented `uv`
  command — only `highspy` was added, no other pin moved; pip-audit clean. Coordinator-side only;
  workers are unaffected.
- pulp 3 and 4 also differ in API (`LpVariable(...)` vs `problem.add_variable(...)`, int status vs
  stats object). `milp_offline.py` loads pulp as `Any` and uses a tiny compatibility layer; both
  versions are exercised (gate on 3.13, suite on 3.10) and give identical results.

## 2026-10-15 — Phase 9: performance smoke budgets
- Scheduler decision, 20 workers × 200 ready tasks: `tesar` took ~14.6 ms (budget 20 ms) because
  every candidate worker recomputed the same chunk geometry. `SchedulingContext.wire_bytes` is now
  memoized per round (the geometry does not depend on the worker): ~7.3 ms.
- Protocol, 10 MB: framing + SHA-256 + array (de)serialization ≈ 22 ms. zlib at level 1 adds
  ≈ 180 ms per 10 MB of image data on this laptop, so with compression the round trip sits right at
  the 200 ms budget. The perf test measures the protocol with `compression = none`; compression is
  a separate codec cost, and real per-chunk results are well under 1 MB.

## 2026-10-15 — Phase 10: verification design choices
- **Optimistic execution, verified completion.** Dependents use unverified results (a VERIFYING
  task satisfies its dependents), but the job only finishes when `final` can run **and** no check
  is outstanding. Because stage 3 runs at the very end, a wrong result reaches the output only if
  sampling let it through unverified; `wrong_results_reached_final` counts the tiles of the final
  output that differ from the reference.
- **What is re-executed:** the producing attempt's whole chunk (contiguous, so it can be sent as
  one task); only the tiles that attempt actually stored are judged ("claims"). A strike is given
  once per check, not per tile; `strikes_to_quarantine = 2` therefore means two separate caught
  attempts.
- **Every worker that returned the judged wrong hash gets the strike** — the producer and any
  replica or untrusted verifier that agreed with it. That is what catches colluders.
- **Free comparisons:** a cancelled replica that finishes anyway is decoded and compared. Agreement
  counts as verification only under the collusion rule (one side trusted, or `rep ≥ 0.9` with
  ≥ 10 trusted-verified results); otherwise, or on disagreement, a trusted check is forced.
- **Verifier choice:** with `prefer_local_verifier` and a configured local worker, checks wait for
  the trusted local worker even when it is busy; the scheduler dispatches checks before policy work
  every round. Without a local worker, the highest-reputation live non-claimant verifies; untrusted
  disagreement goes to a trusted tiebreak. If no worker can ever run a check (no local worker
  configured and no other live worker), the check is skipped with a `verification_skipped` event
  rather than hanging the job (residual risk, SECURITY_DESIGN.md).
- **Counters:** `verified_ok` counts matches; a separate per-run `trusted_verified` count (matches
  judged by the trusted worker) feeds the collusion rule.
- **Report semantics:** `attempts_*`, `wasted_work_s` and `avg_chunk_tiles` count work attempts
  (primary/replica) only; verification attempts have their own fields (`verifications`,
  `verification_overhead_pct`, `tiebreaks`, `free_replica_matches`, `repaired_tasks`,
  `quarantined`, `detection_latency_s`). `test_round_robin_job_with_two_workers` (Phase 4) asserts
  `attempts_ok == attempts_total == tasks − 1` and keeps passing with that definition.
- **Tests adjusted (names kept, no assertion weakened):** the Phase 5 failure-detection harness runs
  with `verification.enabled=false` so its DONE-state assertions test failure handling in
  isolation; `test_fast_worker_gets_bigger_chunks` counts only primary/replica attempts
  (verification attempts reuse the producer's chunk size, so they say nothing about the policy).
- **New lab fault `--collude-key K`** (requires `--lab-mode`): corruption seeded from K and the task
  input, so workers sharing K return identical wrong results — needed for the collusion test.
- **Security events** `wrong_result` and `result_from_quarantined` added to the audit whitelist; the
  verifier unit tests spy on the real `SecurityLog.emit` so an unknown event type fails a test (the
  first e2e run found this: the stubbed log had hidden it).
- **P10 gate** runs `hetero_malicious6` with `verification.base_prob=1.0`: with the default 5 % a
  rare wrong result may slip through, which experiment E5 measures and a gate cannot assert.
- `test_cancel_queued_and_running` (Phase 3) became intermittent for the same CPU-speed reason as
  the Phase 7 fixes: its "long" task (800 k hash steps) now takes 0.24 s, barely longer than the
  0.2 s before the cancel is sent. The long task is now 4 M steps (~1.2 s, under the 5 M limit);
  assertions unchanged.
- **Local takeover (found by the collusion e2e test):** when every remote worker is quarantined,
  dead or gone and `allow_local` is false, no policy would assign the remaining tasks and the job
  hung. The scheduler now hands ready tasks to the trusted local worker whenever no remote worker is
  ACTIVE (`task_fallback_local`, reason `no_remote_workers`) — a job must never hang (Phase 5 rule).

## 2026-10-16 — Phase 11: dashboard design choices
- **Backend interface.** `mdc/dashboard/web.py` talks to the coordinator only through a small
  `Backend` protocol (`state`, `state_update`, `start_job`, `control`, `output_png`, `report`);
  `backend.py` implements it for the real coordinator. The whole security layer is therefore
  tested against a fake backend, plus one integration test against a real coordinator.
- **Login page** is served by the backend as fixed HTML with a plain `<form method=post>` — no
  JavaScript is needed to log in, and nothing from the request is reflected into the page (the
  "wrong token" message is a fixed string selected by `?failed=1`). `/login` accepts the form or
  JSON; both require the dashboard's own `Origin` (login CSRF) and count against the per-address
  rate limit even when the token is right (S.9 item 2).
- `GET /` without a session redirects to `/login`; every other protected route answers 401 JSON.
- **Websocket**: same session cookie, and its `Origin` must be the dashboard's own (otherwise any
  site the operator visits could open a cross-site websocket with the cookie). Events are queued
  per socket (max 1000, oldest-first delivery, extra events dropped for a slow browser) and the
  compact state is pushed every 0.5 s.
- **Jobs from the browser** change the coordinator's configuration in place (policy, α, image,
  iterations, verification, adaptive chunking) — one job at a time, 409 otherwise. Images are
  chosen by basename from `data/images` (regular files only, no symlinks, resolved parent must be
  that directory, `.png/.jpg/.jpeg`).
- **Demo controls** need `security.lab_mode` **and** `dashboard.demo_controls` (403 otherwise);
  the worker itself additionally needs `--lab-mode --allow-demo-control` to obey (Phase 3), so the
  trusted local worker always ignores them.
- The admin-token file constant is called `ADMIN_FILE` (ruff S105 flags any constant whose name
  contains "token"); tests use a random token per run instead of a literal for the same reason.
- aiohttp 3.14 warns on string request keys (an error under our warning filters): the session is
  stored under a typed `web.RequestKey`.

## 2026-10-16 — Phase 12: packaging and calibration
- `scripts/package_worker.sh` is a thin wrapper around `scripts/package_worker.py`: the packager
  is Python so it can be unit-tested, and it builds a **deterministic** zip (sorted entries, fixed
  timestamps and modes). The same sources always give the same SHA-256, so the hash printed on the
  laptop is the one `sha256sum -c` checks on the phone. The build refuses files containing a join
  code or private-key pattern.
- `report.json` per worker gained `compute_s` and `transfer_s` (measured: clamped `compute_ms`,
  and the rest of the round trip minus queueing) so that energy can be calibrated.
  `calibrate-energy` fits `measured − idle_W·makespan ≈ p_compute·compute_s + p_tx·transfer_s` per
  device class by non-negative least squares (exact for two unknowns), needs at least 2 samples per
  class, prints suggestions only and never edits configuration.
- `python -m mdc.coordinator --diag` creates the certificate if it does not exist yet (to show the
  fingerprint workers must pin), like the first coordinator start would. It never prints join codes,
  device secrets or the dashboard token (test `test_coordinator_diag_prints_no_secrets`).
- The physical acceptance step (two phones, one walked out of range) cannot be run here. The
  checklist is in `docs/DEPLOYMENT_TERMUX.md` §8 and in PROGRESS.md.

## 2026-10-16 — Phase 13: experiment framework choices
- **Matrix keys** are configuration keys or `cluster.<flag>`. The latter becomes a worker option
  for every worker through the new `local_cluster.py --extra-args` (lab-mode profiles only), which
  is how E3 varies the drop probability.
- **Isolation per run:** fresh port and a fresh temporary state directory per run (enrollment,
  certificate), so runs never share credentials or devices; seeds `seed + rep` for the coordinator
  and the cluster. Runs are executed repetition-major, so an interrupted experiment has balanced
  partial results; resume skips runs with a `report.json`.
- **Local baseline:** `local_baseline: true` adds one `policy=local` run per repetition, and plots
  then report a *measured* speedup per repetition (else the report's estimated speedup).
- **CI:** mean ± t(0.975, n − 1)·s/√n; with n = 1 the half-width is undefined and no bar is drawn;
  failed runs are excluded from metric means but counted (and count as incorrect in E5's
  correct-output rate).
- **E7** runs ordinary small jobs and solves the MILP afterwards on each saved `snapshot.json`
  (first 24 tiles), instead of solving during the run.
- The first full results (145 runs) are copied to `docs/results/` with a short honest reading; the
  raw `runs/` directory stays out of git.

## 2026-10-16 — Phase 14: final checks
- **Gate bug fixed:** the soak step ran `pytest -m soak -p no:timeout`. Disabling the timeout
  plugin unregisters the `timeout` marker, so `--strict-markers` failed collection of every file
  using it. The soak test sets `@pytest.mark.timeout(0)` itself, so the gate now runs it without
  that flag.
- **Soak test:** 40 consecutive jobs (default; `MDC_SOAK_ROUNDS`) on one coordinator with honest,
  dropping, flaky and lying workers that are randomly killed and restarted. Every job must be
  correct with nothing left over (checks, in-flight attempts, background tasks, events), memory
  growth must stay under 150 MB, and the liar must end quarantined. `verification.base_prob=1.0`
  so correctness is a hard requirement (sampling is E5's subject).
- **"README quick start in a temp clone" (G.3 row 14)** is `PHASE_CHECKS[14]`. It copies what
  git would ship (`git ls-files --cached --others --exclude-standard`: the working tree as it
  would be committed, without `.venv` and `runs/`) into a temporary directory with a fresh HOME and
  runs the README's three commands there. The only additions are `--headless --exit-after-job`
  and free ports, so it cannot collide with a coordinator the user is running. It uses the
  existing virtual environment instead of re-installing the locked dependencies (no network in
  the gate); the install commands themselves are exercised by CI.
- **Code polish:** placeholder docstrings removed; `tests/test_code_quality.py` keeps every
  module documented and every public function typed; `compileall` is clean; no TODO/FIXME in the
  code base (the forbidden scan enforces this in security-critical modules).

## 2026-10-16 — Dashboard login failed in a real browser (`{"error":"origin"}`)
- **Found in manual testing** on first use: pasting the correct token returned `{"error":"origin"}`.
  Spec S.9 item 5 asks for `Referrer-Policy: no-referrer`, but under that policy browsers serialize
  the `Origin` of the page's own POST requests (the login form, every `fetch`) as `null`, and S.9
  item 3 (Origin must equal the dashboard origin) then rejects them. The two rules conflict in
  real browsers. Our tests set `Origin` by hand, so they could not see it.
- **Fix:** `Referrer-Policy: same-origin`. The browser sends the real `Origin` to the dashboard
  itself, and no referrer at all to any other site (the page loads nothing external anyway), so
  the item's intent is kept. `Origin: null` is still refused (new test
  `test_null_origin_is_still_refused`).
- Lesson: security headers interact; the dashboard needs a real-browser check (manual check in
  PROGRESS.md).
