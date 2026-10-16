# Architecture

## 1. Components

```
                 ┌───────────────────────── Coordinator (laptop) ─────────────────────────┐
 Browser ◄─ws──► │ Dashboard (aiohttp :8080)                                              │
                 │      ▲ events                                                          │
                 │ Job (DAG) ◄──► Scheduler ──► Policy (local/rr/fastest/reliable/tesar)  │
                 │      ▲             │  uses: NetEstimator, Reputation, Energy,          │
                 │      │             ▼        Chunking, Registry                         │
                 │ Assembler ◄── Dispatcher ──► Verifier (spot checks, quarantine, taint) │
                 │                   ▲   FailureDetector (ping/pong, timeouts)            │
                 │ TCP server :50010 ┘   UDP beacon → :50000 (broadcast + multicast)      │
                 └──────────────▲─────────────────────────────────────────────────────────┘
                                │ TLS/TCP (worker-initiated, pinned cert, mutual auth)
          ┌─────────────────────┼─────────────────────┬────────────────────────┐
     Local worker (trusted)   Phone A (Termux)      Phone B (Termux)       Laptop C
```

| Module | Responsibility |
|---|---|
| `common/protocol.py` | MDC1 frames: fixed prefix + JSON header + payload; array packing (zlib, SHA-256); strict limits |
| `common/tls.py`, `auth.py`, `credentials.py` | TLS contexts and certificate pinning; mutual HMAC handshake bound to the TLS channel; join codes |
| `common/validation.py` | schema validation of every message, clamping of self-reported values |
| `tasks/` | deterministic integer kernels (blur + edge detection per tile, row statistics, synthetic hash chains), exact tiling with halos |
| `worker/` | the worker: discovery, connect/auth/reconnect loop, task executor threads, battery providers, lab-mode faults and network shaping |
| `coordinator/server.py` | accepts connections, handshake, connection/IP/auth limits, session bookkeeping, probes |
| `coordinator/registry.py` | `WorkerRecord` per enrolled device and its state machine |
| `coordinator/job.py`, `job_builder.py` | the task DAG, attempts, epochs, invalidation |
| `coordinator/dispatcher.py` | turns assignments into frames, validates and accepts results, timeouts, cancellation, energy ledger, verifier hooks |
| `coordinator/scheduler.py` | the scheduling loop: verification checks first, energy gate, policy call, re-validation of every assignment, local fallback/takeover |
| `coordinator/policies/` | `local`, `round_robin`, `fastest`, `most_reliable`, `tesar`, offline `milp_offline` |
| `coordinator/netestimator.py`, `reputation.py`, `energy.py`, `chunking.py` | the models the policies read (bundled as `Estimators`) |
| `coordinator/verifier.py` | spot checks, tiebreaks, strikes, quarantine, retroactive repair, collusion rule |
| `coordinator/failure_detector.py` | PING/PONG, SUSPECT/DEAD, attempt-timeout sweeps, battery samples |
| `coordinator/assembler.py`, `metrics.py` | stage 3 in-process, reference comparison, `report.json` |
| `dashboard/` | REST + websocket UI with its own security layer (`auth.py`, `headers.py`) |

## 2. Data flow of one task

```
 ready tile ──► Scheduler.schedule_once()
                 ├─ Verifier.dispatch()          (pending checks go first)
                 ├─ energy gate (battery floor, budget)
                 ├─ Policy.assign(ctx) ──► Assignment(chunk, worker, purpose, predicted_s)
                 └─ re-validate (capacity, runnable, energy, no 2 copies on 1 device)
                        │
                        ▼
 Dispatcher.send(): Attempt(id, epochs, predicted_s, timeout, energy) ─ TASK_ASSIGN ─► worker
                                                                                      │ compute
 Dispatcher.on_result() ◄─────────────────────────────── TASK_RESULT (sha256, timings) ┘
   ├─ owner + session check; quarantined sender ignored
   ├─ decode: integrity, schema, shape; epoch still current?
   ├─ purpose verify/tiebreak ─► Verifier.on_verify_result() (judge, strike, swap, quarantine)
   └─ purpose primary/replica ─► _accept(): Job.mark_done (first result wins), reputation,
                                 speed + bandwidth + energy updates, cancel other copies,
                                 Verifier.on_accepted() (probation / sampling ─► new check)
                        │
 dependents become READY (optimistic: unverified results count) … all rows done
                        │
 Coordinator: final ready AND no verification outstanding ─► Assembler.finish() (stage 3,
 output.png, compare with reference) ─► report.json
```

If a check later proves a tile wrong, `Job.replace_result` swaps in the trusted result and
invalidates its row statistics (and `final`). Epochs make sure that no in-flight attempt computed
from the wrong tile can deliver a stale result.

## 3. State machines

**Worker (`registry.py`)**

```
CONNECTING ─auth─► PROBING ─probe ack/3 s─► ACTIVE ◄─pong─ SUSPECT ─silence─► DEAD ─reconnect─► CONNECTING
     │                                         │  └──silence──►┘                  ▲
     └──────────── (any state) ── strikes ≥ 2 ─┴─────────────► QUARANTINED (sticky)│
                                     BYE / revoked ──► LEFT ───────────────────────┘
```

Only ACTIVE workers get work. QUARANTINED has no outgoing transitions, so reconnecting does not
lift it.

**Task (`job.py`)**

```
PENDING ─deps satisfied─► READY ─attempt─► RUNNING ─result─► DONE ◄─match── VERIFYING
   ▲                        ▲                 │                │  └─check created─►┘
   │                        └──attempt failed─┘                │
   └────── a dependency invalidated ──────────── invalidate() ─┴─► INVALIDATED (runnable again)
```

**Attempt:** `inflight → ok | error | timeout | lost | cancelled | late | late_accepted |
rejected`. A timed-out attempt whose result still arrives while its task needs it becomes
`late_accepted`.

## 4. Concurrency model

* **One asyncio event loop** per process. The coordinator's server, dispatcher, scheduler,
  failure detector, beacon and dashboard all share it, so there is no shared-memory locking.
* **CPU-bound and blocking work runs in executors:** kernels on the worker (thread pool sized by
  `--slots`; numpy releases the GIL), image loading, the reference image, stage 3 and PNG encoding
  on the coordinator.
* **Per connection:** one reader task and an `asyncio.Lock` around writes. Every network read has
  a timeout or runs in a task the failure detector can cancel.
* **Scheduler:** a single loop woken by an event whenever something changes, plus a safety tick.
  It uses the cancellation-safe `timeutil.wait_event` (Python 3.10's `asyncio.wait_for` can
  swallow a cancellation, see DECISIONS.md).
* **Background tasks** are tracked and cancelled on shutdown. The test suite fails if a task is
  destroyed while pending.
* **Clocks:** only the coordinator's monotonic clock is used for RTT, timeouts and energy
  integration. Workers' timestamps are echoed or checked, never trusted.

## 5. Key design decisions

1. **Workers connect to the coordinator.** Only the laptop needs an open inbound port. Phones
   behind restrictive networks can make outbound connections, which also simplifies hotspot and
   NAT setups.
2. **Discovery is a convenience.** Beacons (broadcast and multicast) let workers find the laptop
   automatically. When a network blocks them, `--coordinator HOST:PORT` works; the pinned
   fingerprint still protects the connection.
3. **One persistent TCP connection per worker** carries control (PING/PONG, CONTROL, CANCEL) and
   data (tasks, results). Framing keeps messages apart, and a per-connection lock serializes
   writes.
4. **The coordinator owns all clocks** (see §4): no clock synchronization is needed.
5. **The local worker is a real subprocess** over loopback, using the same code path as remote
   workers. It is `trusted`: the ground truth for verification, the fallback for exhausted tasks,
   and the `local` baseline.
6. **Every execution is an Attempt** with a unique id. Retries, replicas and verifications are
   separate attempts. The first accepted result wins, and duplicates are compared or ignored.
7. **Security boundary:** the coordinator trusts only the local worker. Every remote worker may
   be compromised (valid credentials, wrong answers), and every network peer may be an attacker.
   Workers trust only a coordinator that presents the pinned certificate **and** proves knowledge
   of their device secret. See `SECURITY_DESIGN.md`.
8. **Optimistic execution, verified completion.** Dependents start from unverified results for
   speed; the job finishes only when no check is outstanding, and taint propagation repairs any
   damage.
9. **Safety rules live in the scheduler, preferences in policies.** The battery floor, energy
   budgets, capacity, the replica rule and the local takeover hold whatever a policy returns.
10. **Estimators are shared objects** (`Estimators`: network, reputation, energy) read by
    policies and written by the dispatcher and failure detector, and summarized in `report.json`
    and `snapshot.json`.

Phase-by-phase choices and deviations from the specification are recorded in `DECISIONS.md`.
