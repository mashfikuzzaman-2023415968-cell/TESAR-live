# Security design

This document expands spec Section S. It is completed in Phase 1 (transport, authentication,
enrollment, validation) and kept up to date in every later phase. The short version, including
residual risks, is in [../SECURITY.md](../SECURITY.md).

## Assets

1. Device secrets (per-worker HMAC keys) and the coordinator's TLS private key.
2. The dashboard admin token and sessions.
3. Correctness of job results.
4. Confidentiality of the user's input images in transit.
5. Availability of the coordinator and of each phone (battery, CPU, storage).
6. The worker devices themselves.

## Controls implemented so far

### Phase 0 — configuration and local secrets

* **Strict configuration** (`mdc/common/config.py`): unknown keys, wrong types (a `bool` is
  never accepted as a number), non-finite numbers and out-of-range values are rejected.
* **Insecure combinations refused (fail closed):** `security.tls: false` requires the explicit
  `--insecure-dev` flag *and* a loopback bind address; `dashboard.demo_controls` requires
  `security.lab_mode`.
* **Private state directory** (`mdc/common/statedir.py`): created `0700`, tightened if looser,
  refused if it is a symlink or owned by another user. Secret files are written atomically
  (`mkstemp` → fsync → `os.replace`) with mode `0600`, and secret files readable by group/others
  are refused with instructions to `chmod 600`.
* **Repository hygiene:** `.gitignore` covers state, join files, keys and run outputs;
  `scripts/check_forbidden.py` (pre-commit + gate) rejects committed join codes, PEM private
  keys and banned code constructs; `scripts/check_worker_isolation.py` keeps phone-side code
  free of laptop-only dependencies.
* **Supply chain:** hash-pinned lock files (`requirements.lock`, `requirements-worker.lock`),
  `pip-audit` and `bandit` in the gate.

### Phase 1 — transport, authentication, enrollment, validation, secrets

#### Threat model

| ID | Adversary | Capabilities | Controls (module) |
|---|---|---|---|
| A1 | Passive LAN eavesdropper | sniffs WiFi | TLS ≥ 1.2 on every worker connection (`common/tls.py`) |
| A2 | Active LAN attacker | spoofs beacons, MITM, impersonates either side, floods, malformed frames | certificate pinning, mutual HMAC auth with channel binding (`common/auth.py`), enrollment (`coordinator/enrollment.py`), strict framing/validation (`common/protocol.py`, `common/validation.py`); rate limits and timeouts in Phase 4 |
| A3 | Malicious enrolled worker | valid credentials, lies | clamping of self-reported values (`validation.clamp_*`); attempt ownership (Phase 4); verification + quarantine (Phase 10) |
| A4 | Malicious coordinator (phone's view) | crafted tasks/CONTROL | task-kind whitelist + local limits (`validate_task_params`, `validate_task_input`); CONTROL lab-mode gating (Phase 3) |
| A5 | Web attacker | CSRF/XSS/rebinding | Phase 11 |
| A6 | Other local users | read files | 0700 state dir, 0600 key/registry/join files, secrets never in argv/logs |
| A7 | Supply chain | bad dependency | hash-pinned locks, pip-audit, bandit |

#### Data flow of trust

```
  operator ── mdc.cli enroll ──► devices.json (0600: worker_id, secret)
      │                               │
      └── join code (QR / USB) ──► phone: host, port, worker_id, secret, cert fingerprint
                                      │
  phone ── TLS ──► coordinator: cert pinned (fingerprint) ─► CHALLENGE / HELLO(auth) / WELCOME(proof)
```

#### Transport (S.3)

* Coordinator certificate (`coordinator/certs.py`): ECDSA P-256, self-signed, valid
  `security.cert_days`; `state_dir/tls/cert.pem` (0644) and `key.pem` (0600). Loading refuses a
  loose key file, a key that does not match the certificate, or an **expired** certificate (it is
  not silently replaced: that would break every pin without telling anyone). Rotation:
  `mdc.cli rotate-cert --yes [--reissue-dir DIR]`.
* Server context: `PROTOCOL_TLS_SERVER`, minimum TLS 1.2, `OP_NO_COMPRESSION`, default ciphers.
* Client context: minimum TLS 1.2, `OP_NO_COMPRESSION`, `check_hostname = False`,
  `verify_mode = CERT_NONE` **because trust comes from exact-certificate pinning**:
  `open_pinned_connection` hashes the presented DER certificate and compares it with the pinned
  fingerprint (`hmac.compare_digest`) before returning; on mismatch it closes the connection
  without sending a byte. Pinning one exact certificate is stricter than CA validation — even a
  CA-signed certificate for the right name is rejected. `common/tls.py` is the only module
  allowed to touch these settings (`check_forbidden.py`).
* Tested (`tests/security/test_sec_tls.py`): pin ok; wrong pin aborts before application data;
  an impostor certificate with the same name is rejected; a plaintext client never reaches the
  handler; TLS 1.0/1.1 clients are refused; a server that never completes TLS hits the timeout.

#### Mutual authentication (S.3.5)

`auth = HMAC-SHA256(secret, "MDC-W" | nonce_c | nonce_w | worker_id | fp)` and
`proof = HMAC-SHA256(secret, "MDC-C" | nonce_w | nonce_c | worker_id | fp)`, with every field
length-prefixed. Fresh 16-byte nonces from `secrets`, 32-byte secrets, constant-time comparison.
Tested: replay, reflection, wrong fingerprint (MITM), wrong/rotated/unknown/revoked credentials,
rogue coordinator without the secret, non-HELLO first message (`test_sec_auth.py`).

#### Enrollment and credentials (S.4)

* No self-registration, no shared token: `mdc.cli enroll --name N` creates a `worker_id`
  (`w_` + 64 random bits) and a 32-byte secret per device. Names `^[A-Za-z0-9_.-]{1,32}$`,
  unique among active devices; at most 256 records.
* `devices.json` (0600) is changed only under an exclusive lock file and written atomically;
  it is parsed strictly (exact field sets, valid ids/names/secret lengths, no duplicate ids or
  active names) and refused if group/world accessible.
* Join code `MDC1.` + base64url(JSON) — strict decoding (≤ 1 KB, exact field set, every field
  validated). Join files are 0600; a group/world-readable join file is refused with a `chmod 600`
  hint. The worker gets the code from a file or `MDC_JOIN_CODE`, never from argv.
* `revoke` (secret stops working; a running coordinator reloads the registry on mtime change —
  `reload_if_changed`, wired up in Phase 4), `rotate-secret` (same id, new secret),
  `list-devices` (no secrets), `show-fingerprint`.

#### Input validation (S.5)

Framing checks lengths before reading; strict JSON (no NaN/Infinity/overflow/huge ints/duplicate
keys/depth > 8). One explicit validator per message type; payload rules per type; arrays:
dtype whitelist, ≤ 3 dims of 1..16384, `raw_len` consistent and ≤ 64 MiB, bounded
decompression, then SHA-256. Task kinds are a whitelist; parameters are checked against
`max_task_params`; per-kind result validators (histogram sums, digest format, output shape).
Fuzzing (`tests/property/`) shows decoders/validators raise only the declared error types,
never hang and keep peak memory bounded.

#### Secrets in logs and files (S.10)

* `common/redact.py`: dict keys that name secrets → `***`; join codes and PEM private keys are
  masked inside any string. Applied to every log record **at creation time** (a global
  `LogRecordFactory`, so even handlers we did not install see only redacted records), to log
  handlers as a filter, to config dumps and to the security log.
* Peer strings are sanitized before logging (`safe_names.sanitize_for_log`: control/format
  characters removed, 256-char cap). Payload bytes are never logged.
* `coordinator/security_log.py` writes `state_dir/logs/security.jsonl`: whitelisted event types,
  redacted + sanitized fields, rotation (5 MB × 5), per-type rate limiting with a
  `suppressed_before` count.

### Phase 2 — task safety (worker side)

* **No code over the network** (S.7.1): `mdc/tasks/base.py` `TASK_KINDS` is the complete list of
  computations a worker can run (`img_tile`, `row_stats`, `synthetic`). Unknown kinds are refused
  before any parameter is looked at.
* **Limits inside the kernels** (S.12 row 2): every kernel checks its parameters against
  `TaskLimits` with exact `int` type checks, even when called directly; `execute()` additionally
  validates the input array (2-D `uint8`, ≤ `max_tile_pixels`, larger than 2·halo) and validates
  its own output before returning.
* **CPU-time guard** (S.6.5): kernels check a cooperative deadline between blur iterations and
  every 10 000 hashes, and stop with `error: "time_limit"`.
* **Safe image loading** (S.9.8, coordinator): Pillow `MAX_IMAGE_PIXELS = 50 MP`,
  `DecompressionBombError` caught, images larger than 8192 × 8192 refused.
* Tested in `tests/security/test_sec_kernels.py` (hostile kinds, parameters and inputs; stricter
  worker limits win; deadlines).

### Phase 3 — protecting worker devices (S.7) and trusting the network (S.3.7)

| Control | Where | Test |
|---|---|---|
| TLS + exact certificate pin before any byte is sent | `Worker._connect` → `open_pinned_connection` | pin mismatch: the stub never sees a HELLO |
| Coordinator must prove the device secret (WELCOME `proof`) | `Session.handshake` | wrong proof → never served, `coordinator_proof_failed` logged |
| Credentials only from a 0600 join file, stdin (`--join-code -`) or `MDC_JOIN_CODE` — never argv | `mdc/worker/__main__.py` | loose file refused; `--join-code MDC1...` refused |
| Fault flags and `--allow-demo-control` require `--lab-mode` (red banner) | `check_safety` | every combination refused without it |
| CONTROL ignored and security-logged unless `--lab-mode --allow-demo-control` | `Session._on_control` | crash/leave/set_* ignored, `control_ignored` × 3 |
| Trusted `--local` worker: no faults, no CONTROL, loopback coordinator only | `check_safety` | refused otherwise |
| Worker's own task limits; the coordinator can only tighten them | `stricter_limits` | loose WELCOME limits do not loosen |
| Bounded queue (`slots + prefetch + 1`), `busy` beyond it | `Session._on_assign` | third task → `busy` |
| Battery floor (`--min-battery`, default 15 %) unless charging | `battery_allows_work` | `battery_low`; charging bypass |
| Schema violation after authentication → disconnect; > 10 unknown messages → disconnect | `Session._reader_loop` | `validation_failed`; 11th unknown message |
| No listening TCP socket; only inbound socket is the UDP beacon listener | worker design | **real process** checked in `/proc/net/tcp*` |
| Only one external command (`termux-battery-status`): resolved once via `shutil.which`, argument list, 5 s timeout | `battery.TermuxBattery` | AST scan: `subprocess` only in `battery.py` |
| Nothing written to disk except a rotating log (5 MB × 3), directory 0700, files 0600 | `PrivateRotatingFileHandler` | real process: only `~/.mdc-worker/logs/worker.log` created |
| Refuse to run as root unless `--lab-mode --allow-root` | `check_safety` | `geteuid() == 0` refused |
| Beacons untrusted: ≤ 1024 bytes, ≤ 10/s, strict JSON, fingerprint must equal the pin | `discovery_listener.parse_beacon` | spoofed fp / oversize ignored |
| `--diag` prints identity and fingerprint only, never the secret or join code | `run_diag` | output scanned for every encoding of the secret (also a gate check) |

Worker security events are written as `SECURITY <event>` lines to the worker's own log
(`pin_mismatch`, `coordinator_proof_failed`, `validation_failed`, `control_ignored`).

Residual risk specific to workers: a lab-mode worker deliberately misbehaves; lab mode must never
be enabled on devices that participate in real work (the banner says so on every start).

### Phase 4 — the coordinator's server (S.6, S.8, S.4.5, S.4.7)

| Control | Implementation | Test |
|---|---|---|
| TLS handshake bounded by `handshake_timeout_s`; then CHALLENGE/HELLO/WELCOME within the same budget | `asyncio.start_server(ssl_handshake_timeout=…)` + `wait_for(_handshake)` | slowloris (no TLS) and silent-after-TLS peers dropped |
| `max_connections`, `max_connections_per_ip` | counters in `Server._handle` | 4th connection / 3rd from one IP closed, `connection_limit` logged, slots released on close |
| Auth-failure rate limit: `auth_fail_limit` failures in `auth_fail_window_s` → ban `auth_ban_s`; bounded LRU of 1024 IPs | `common/ratelimit.FailureTracker` | ban starts, even a correct credential is refused while banned, recovery after expiry |
| Idle timeout after authentication | `wait_for(read_frame, idle_read_timeout_s)` | half a frame then silence → closed |
| Oversize/malformed frames → close (`frame_rejected`); schema violations → close (`validation_failed`); > 10 unknown messages → close | `Server._serve` | each case |
| Attempt ownership: a result is accepted only from the session the attempt was sent to | `Dispatcher.on_result` | thief's result rejected (`result_from_wrong_worker`), counted as its failure; owner's result still accepted |
| Unsolicited results: > 3 per minute → disconnect + strike | `SlidingCounter` per connection | 4 fake attempt ids → closed, strike recorded |
| Result validation before acceptance: SHA-256 (transport), shape/dtype, per-kind JSON schema, row pixel count, clamped timings, stale-epoch check | `Dispatcher._extract_output` | error, wrong shape, integrity, JSON-for-image, lying row stats, absurd `compute_ms` |
| Trusted local worker: ephemeral in-memory credential via `MDC_JOIN_CODE`, only from loopback, only it may claim `trusted_local` | `LocalCredential`, `trusted_local_decision` | enrolled device claiming trusted → `trusted_local_spoof`; local credential not claiming; rule table incl. non-loopback |
| Live revocation / secret rotation closes sessions within one check interval (2 s) | `Server.check_revocations` | revoked and re-keyed devices disconnected, `device_revoked_session_closed` |
| `security.jsonl` audit log, reserved field names refused (no forged event types) | `SecurityLog.emit` | regression test |
| Insecure dev mode logged (`insecure_mode_enabled`), only with `--insecure-dev` and a loopback bind | config + coordinator | e2e in insecure mode |

### Phase 6 — estimates that a worker cannot game (S.8.1, S.12 rows 5–9)

* RTT and probe times are measured on the coordinator's clock from its own send records; the echoed
  `t_c` is only checked for equality (a falsified echo yields no sample). A probe ACK must carry the
  current `probe_id` and the exact `received_bytes`.
* Every estimator input is validated: non-finite or negative values are dropped, RTT samples above
  600 s are rejected, bandwidth is clamped to `[min_bw, 1e9]` B/s, noisy samples (< 5 ms or < 16 KiB)
  are ignored, compression ratios are capped at 2. Worker-reported `compute_ms`/`queue_ms` were
  already clamped to the measured total (Phase 4). Hostile-value tests: `tests/test_netestimator.py`.
* Bounded memory: 64 remembered PINGs; one probe in flight per worker.

### Phase 7 — reputation that a worker cannot inflate

* Reputation is keyed by the enrolled `worker_id` (never by name or address), so reconnecting does
  not reset a bad history within a coordinator run.
* Only coordinator-observed outcomes feed it: accepted results, timeouts measured on the
  coordinator's clock, lost connections, error reports, and (Phase 10) verification mismatches. A
  worker cannot report its own success rate. Results arriving on another worker's attempt count as an
  error for the sender, never as a success.
* The attempt length used for λ is the coordinator's own prediction, not a worker-reported time;
  non-finite or negative values are treated as 0 and λ is clamped to ≥ 1e-6, so R stays in [0, 1]
  (hostile-value and property tests in `tests/test_reputation.py`). Memory is O(workers).

### Phase 8 — battery and energy protection for device owners (S.7.6, S.12 rows 5–9)

* The battery floor is now enforced on **both** sides: the worker refuses tasks below
  `--min-battery` (Phase 3) and the coordinator never assigns to a remote worker below
  `workers.min_battery_pct` unless it is charging, nor beyond its per-job energy budget — whatever
  the policy returns (the scheduler re-checks every assignment).
* Self-reported battery values are untrusted: `pct` is clamped to 0–100 on receipt (Phase 4),
  `current_ma` to 20 A, `voltage_v` to 30 V; non-numbers, NaN, ±inf and negative voltages are
  ignored; the integration interval comes from the coordinator's clock and is capped. Lying about the
  battery (or the device class) can only change how much work the liar itself receives.

### Phase 9 — policies and the offline tool

* Every policy only sees ACTIVE workers with capacity; the scheduler re-validates whatever a policy
  returns (known task ids, runnable state, capacity, energy, and no second copy of a task on the
  same device), so a buggy or experimental policy cannot bypass the safety rules.
* `milp_offline` reads a local `snapshot.json` (written by our coordinator, possibly edited by
  hand): size limit 32 MiB, JSON object required, every number checked finite and clamped, unknown
  structure rejected with an error (exit 2), at most 8 devices and `--max-tasks` tiles. It never
  connects to anything.

### Phase 10 — verification, quarantine, collusion resistance (S.3, S.12 row 10)

* **Transport integrity ≠ correctness.** The payload SHA-256 (Phase 1) only proves the bytes were
  not damaged in transit; a dishonest worker hashes its wrong result correctly. Correctness is
  checked by re-execution on the trusted local worker (`verifier.py`).
* **Who is verified:** every untrusted worker's first `probation_tasks` results, then a fraction
  `min(1, base_prob + (1 − rep) · low_rep_boost)` chosen by a seeded `random.Random` (`rng`, not a
  security RNG: the seed is the operator's experiment seed and never leaves the coordinator).
* **Collusion:** two matching untrusted results are not trusted unless one side is trusted or has
  a long trusted-verified record (`rep ≥ 0.9`, ≥ 10 checks); every worker that returned the same
  wrong hash is struck. Hash comparisons use `hmac.compare_digest`.
* **Containment:** after `strikes_to_quarantine` strikes the worker is QUARANTINED — sticky in the
  registry, so reconnecting (new session, same `worker_id`) does not lift it; its in-flight work is
  cancelled and requeued, everything it produced that was not verified is re-executed and dependents
  recomputed, and any later result from it is ignored (`result_from_quarantined`). Revocation
  (`mdc.cli revoke`) remains the way to remove it permanently.
* **Residual risks (to SECURITY.md in Phase 14):** a worker that behaves during probation and lies
  rarely afterwards is caught per lie with probability `p` (≈ 0.05–0.3), so a wrong tile can reach
  the output (measured by `wrong_results_reached_final`, experiment E5); without a local worker and
  with no other live worker a check is skipped (`verification_skipped`); a coalition that controls
  the trusted local machine is out of scope.

### Phase 11 — dashboard (S.9)

| S.9 item | Control | Test |
|---|---|---|
| 1 binding | loopback by default; any other host needs HTTPS with the coordinator certificate, refused at startup otherwise | `test_non_loopback_dashboard_requires_https` |
| 2 authentication | 32-byte admin token in `state_dir/admin_token` (0600, insecure file refused), `hmac.compare_digest`, per-address rate limit, server-side sessions (random 32-byte ids, TTL, max 64), `HttpOnly`/`SameSite=Strict`/`Secure` on HTTPS, logout | `test_login_*`, `test_form_login_sets_a_strict_cookie`, `test_secure_cookie_on_https`, `test_sessions_*` |
| 3 CSRF | POST + JSON only, `X-MDC-CSRF` equal to the session token, `Origin` equal to the dashboard origin (login included) | `test_csrf_and_origin_are_required` |
| 4 DNS rebinding | `Host` must be the configured host, `localhost` or `127.0.0.1` with the port | `test_bad_host_is_rejected` |
| 5 headers | CSP (no `unsafe-inline`/`unsafe-eval`, no external sources), nosniff, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`, COOP, `Cache-Control: no-store` on the API | `test_security_headers_and_csp` |
| 6 XSS | DOM built with `createElement`/`textContent` only; no inline scripts or handlers; scanner rules | `test_frontend_has_no_dangerous_sinks_or_inline_code`, `check_forbidden.py` |
| 7 path traversal | images by basename from a server-generated list, symlinks excluded, resolved parent check, suffix check, no upload | `test_path_traversal_and_symlinks_rejected` |
| 8 image parsing | `MAX_IMAGE_PIXELS = 50 M`, decompression-bomb error caught, > 8192² refused (job builder, Phase 2) | existing job-builder tests |
| 9 errors | generic JSON errors, exceptions only in the coordinator log, `/healthz` = `{"ok": true}`, 16 KiB body limit | `test_errors_never_leak_details` |

The websocket requires the session and the dashboard's own `Origin` (cross-site websocket
hijacking); demo controls need `lab_mode` and `demo_controls` on the coordinator **and**
`--lab-mode --allow-demo-control` on the worker.

### Phase 12 — distribution to phones

* The worker zip holds only worker-side code (no coordinator, dashboard, tests, bytecode or
  credential files), is built reproducibly, and the build fails if any file matches a join-code or
  private-key pattern. Its SHA-256 is published next to it and checked on the phone
  (`sha256sum -c`) before unzipping.
* Plain HTTP is used **only** for that public zip, and the server is stopped afterwards. Join codes
  travel by QR scan or USB, are stored with mode 0600 (the worker refuses anything looser), and can
  be revoked instantly (`mdc.cli revoke`).
* Both `--diag` commands print public information only (fingerprints, addresses, counts); tests
  check that no join code or token appears.
