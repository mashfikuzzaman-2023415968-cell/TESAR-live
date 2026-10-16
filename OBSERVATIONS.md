# Observations from real-world testing

Real-world problems found while testing on real devices and networks are research material:
they show where the paper's model does (or does not) capture reality. Add one entry per
problem using the template below.

## Template

```
## <date> — <short title>
- Environment: (network name/type, devices, OS versions)
- What happened:
- How we detected it:
- Fix/workaround:
- Research relevance (does the paper's model capture this?):
```

## Entries

## 2026-10-07 — PONG latency rises while the worker computes (laptop, loopback)
- Environment: one laptop (Linux, Python 3.13), stub coordinator and worker over TLS loopback.
- What happened: during a 1.57 s SHA-256 hash-chain task the PING→PONG round trip rose from
  ≈ 1.6 ms (idle) to 26–32 ms, then fell back as soon as the task finished.
- How we detected it: the stub coordinator's manual check prints every PONG's round-trip time.
- Fix/workaround: none needed for liveness (far below the 3 s suspect threshold). The cause is the
  GIL: hashing 32-byte blocks does not release it, so the event loop only runs between the
  interpreter's switch intervals. numpy-heavy image tasks release the GIL and disturb PINGs less.
- Research relevance: RTT samples taken while a worker computes over-estimate the *network* delay.
  Phase 6 should prefer the minimum RTT (or idle-time samples) when separating transfer time from
  RTT; the paper's model treats communication and computation delay as independent, which this
  shows is only approximately true on a single device.

## 2026-10-07 — Plugged-in laptop reported as a dying battery (development laptop)
- Environment: Linux laptop on AC power, battery charge thresholds 50 %/90 %; sysfs reported
  `BAT0/capacity = 2`, `status = "Not charging"`, `AC/online = 1`.
- What happened: the trusted local worker (battery provider `auto` → sysfs) mapped "Not charging"
  to `charging = false`, saw 2 % < 15 %, and refused every task with `battery_low`. The coordinator
  then re-sent tasks in a tight loop (~10 000 attempts/min) and the job never finished.
- How we detected it: Phase-5 tests with the local worker started timing out; `events.jsonl`
  showed thousands of `battery_low` errors.
- Fix/workaround: (1) "charging" now means *externally powered*: any online `Mains` supply counts
  (Termux already used `plugged`); (2) the trusted local worker runs with `--min-battery 0`;
  (3) the coordinator backs off exponentially from workers that return errors.
- Research relevance: the paper's energy model assumes a battery level that maps to available
  energy. Real devices add charge thresholds, battery-care modes and mains power, so "battery %"
  alone is a poor eligibility signal; whether the device is externally powered matters more.

## 2026-10-08 — Bandwidth estimates under one-directional shaping (laptop, emulated links)
- Environment: laptop, loopback; one worker shaped to 1.5 Mbit/s + 20 ms on its sends only.
- What happened: the estimator ranked the shaped worker below the unshaped one (correct order) but
  estimated ~3 MB/s instead of ~0.19 MB/s; the unshaped worker's estimate also drifted down from
  11 MB/s to 2.8 MB/s when coordinator and workers shared one Python process.
- How we detected it: per-attempt `est_bw_bytes_per_s` in `events.jsonl` (Phase-6 test).
- Fix/workaround: chunk-size behaviour is tested with separate processes; the formula is kept as
  specified and the limitation documented (DECISIONS.md).
- Research relevance: transfer-time models assume symmetric links and that measured time is network
  time. Asymmetric links (typical for phones: uplink much slower than downlink) and host-side delays
  (GIL, busy event loops, power-saving radios) violate both; a per-direction estimate would be a
  natural improvement.

## 2026-10-15 — Online greedy vs offline optimum at job start (laptop, hetero_faulty6)
- Environment: laptop, 6 emulated workers + local, `tesar` with α = 0.5, κ = 10.
- What happened: every remote device starts at reputation 0.5, so `tesar` replicated about half of
  all attempts (17 of 33) to reach the 0.9 reliability target; the offline MILP on the same snapshot
  kept almost all tiles on the trusted local worker (objective 1.21 vs the greedy's 2.90).
- How we detected it: `PHASE_CHECKS[9]` runs `milp_offline` on the run's own `snapshot.json`.
- Fix/workaround: none needed; documented in `docs/TESAR_ALIGNMENT.md` §5 (different
  normalization of the speed term online vs offline).
- Research relevance: the cold-start reputation prior dominates early decisions. How fast trust is
  earned (priors, decay) and how the speed term is normalized decide whether volunteers get work at
  all; the α sweep (E2) and the MILP comparison (E7) should be read with that in mind.

## 2026-10-15 — Cost and speed of catching liars (laptop, hetero_malicious6)
- Environment: laptop, 6 emulated workers (one lies on 30 % of results, one on 5 %) + local,
  `tesar`, every result verified (`base_prob = 1`).
- What happened: the 30 % liar was quarantined in every run, ~0.4 s after its first caught wrong
  result; the 5 % liar only in some runs (it must be caught twice). Verification cost ~16 % of all
  compute time; quarantine re-executed 20–64 already accepted tasks (its unverified work and their
  dependents).
- How we detected it: `PHASE_CHECKS[10]` report fields (`verification_overhead_pct`,
  `detection_latency_s`, `repaired_tasks`).
- Fix/workaround: none; the trade-off is the experiment (E5: `base_prob` vs overhead vs wrong
  results reaching the output).
- Research relevance: reputation alone cannot detect a lying device — a wrong result looks like a
  success until someone recomputes it. The paper's reliability model assumes failures are visible;
  silent corruption needs an explicit, costly verification term.

## 2026-10-16 — On one laptop, offloading does not pay (laptop, all experiments)
- Environment: laptop, 5–6 emulated workers + local worker, jobs of 0.2–10 s.
- What happened: `local` (only the laptop's own worker) finished E1 in 0.25 s, faster than every
  offloading policy (tesar 0.49 s, fastest 1.47 s, round_robin 4.48 s).
- How we detected it: E1 summary (`docs/results/e1_baselines`).
- Fix/workaround: none: emulated workers share the same CPU and only add TLS, serialization,
  verification and coordination. Offloading pays off only when remote devices add real compute,
  i.e. on real phones and with longer jobs (Phase 12 checklist).
- Research relevance: speedup depends on the ratio of task compute to per-task overhead (RTT,
  serialization, verification). The paper's speedup model should include fixed per-offload costs,
  or small tasks will be predicted to benefit when they do not.

_Real-device (phone) observations start in Phase 12._
