# Alignment with the TESAR paper

This project's scheduler is **TESAR-inspired**: it implements the paper's core idea (decide where
each task runs by trading execution speedup against reliability, under device energy limits, with
reliability derived from device reputation and with task dependencies modelled as a graph). The
paper's exact equations are **not** reproduced here. The sections marked *to fill in* are for the
author to complete after reading the paper, so that every match or difference is stated against
the paper's own numbering.

## 1. Paper's problem statement (to fill in)

_Summarize in 3–5 sentences: the system model (devices, network, task graph), what is decided
(task → device mapping), what is optimized, and under which assumptions (static or online, known or
estimated parameters)._

-
-

## 2. Paper's objective and constraints (to fill in, with equation numbers)

| Paper eq. | Meaning | Symbols | Our counterpart |
|---|---|---|---|
| (?) | objective | | `cost(d)` in §3.1 / MILP objective in §3.2 |
| (?) | speedup / execution time model | | `predicted_s` (§3.1) |
| (?) | reliability model | | `R = rep · exp(−λ·T)` (§3.1) |
| (?) | reputation update | | Beta reputation (§3.3) |
| (?) | energy constraint | | energy budget (§3.4) |
| (?) | dependency / precedence constraint | | job DAG (tiles → row stats → final) |
| (?) | other | | |

## 3. Our formulation

### 3.1 Online greedy rule (`mdc/coordinator/policies/tesar.py`)

For the first ready task `t` (stage-2 tasks first, then tiles in row-major order), and every
eligible, energy-feasible device `d` (the trusted local worker only if `scheduler.allow_local`):

```
chunk     = t plus the following tiles of the same row (adaptive chunking for d), n = |chunk|
T_s       = queue_wait(d) + rtt(d) + (bytes_in + bytes_out) / bw(d) + compute(chunk, d)
R         = rep(d) · exp(−λ(d) · T_s)          (0.9999 for the trusted local worker,
                                                0 below the battery floor unless charging)
speed_cost = (T_s / n) / T_loc(t)              (T_loc = compute time of t on the local worker)
rel_cost   = −ln(max(R, 1e−6)) / n
cost(d)    = α · speed_cost + (1 − α) · κ · rel_cost
```

Pick `d* = argmin cost(d)`. If replication is on and `R(chunk, d*) < reliability_target`, add copies
of the same chunk on the next-cheapest other devices until `1 − Π(1 − R_i) ≥ target` or
`max_replicas` extra copies. The first accepted result wins; the others are cancelled.

`−ln R` turns the job-level reliability (a product over tasks) into a sum, the same shape as time,
so both terms add up in one cost. `α = 1` gives the `fastest` policy's choice, `α = 0` a per-tile
`most_reliable`. Defaults: `α = 0.5`, `κ = 10`, target 0.90, at most 2 replicas.

### 3.2 Offline MILP (`mdc/coordinator/policies/milp_offline.py`)

Over the stage-1 tiles of a run's `snapshot.json` (one tile per task, no chunks, no replicas):

```
variables  x[t,d] ∈ {0,1},  M ≥ 0
Σ_d x[t,d] = 1                         ∀t
Σ_t x[t,d] · c[t,d] ≤ M · slots[d]     ∀d       (c = rtt + bytes/bw + compute)
Σ_t x[t,d] · e[t,d] ≤ budget[d]        ∀d       (e = p_compute·compute + p_tx·transfer)
minimize  α · M / M_ref + (1 − α) · κ · Σ x[t,d] · (−ln R[t,d]) / N
```

`M_ref` is the total local compute time, `N` the number of tasks. Run it with
`python -m mdc.coordinator.policies.milp_offline --snapshot runs/<run>/snapshot.json --alpha 0.5`;
it also evaluates the greedy rule statically on the same predictions and prints the gap.

### 3.3 Reputation and failure rate (`reputation.py`)

Beta reputation per device: `s, f` start at 1, decay 0.98 per event, success `s += 1`, failure of
kind k `f += w_k` (timeout, disconnect, error 1; wrong result 5); `rep = s / (s + f)`. Failure rate
`λ = −ln(1 − p) / T` from EWMAs of the failed-attempt fraction `p` and the attempt length `T`.

### 3.4 Energy (`energy.py`)

`E = p_compute(class) · compute_s + p_tx(class) · transfer_s`; per-job budget
`capacity · battery% · max_job_energy_pct`; devices below `min_battery_pct` (not charging) are
ineligible; the local worker is exempt.

## 4. Matches (to confirm against §2)

- Speedup against reliability with a single weight (α), reliability from device reputation.
- Energy as a hard per-device constraint.
- Task dependencies as a DAG; a task becomes ready when its inputs are done.
- An optimization model (§3.2) whose objective has the same two terms as the online rule.

## 5. Differences and why

| Topic | Paper (to confirm) | This implementation | Why |
|---|---|---|---|
| Solving | likely an offline optimization | **online greedy** every scheduling round; the MILP is offline only | devices join, leave and slow down during a job; an exact solve per round is too slow and stale |
| Reputation | possibly assumed or given | **measured** from observed outcomes (Beta + decay) | real devices; no ground truth is available |
| Execution time | possibly compute only | **network-aware**: queue wait + RTT + bytes / measured bandwidth + compute | on WiFi, transfer is often the larger part |
| Task granularity | fixed tasks | **adaptive chunks** of tiles, sized per device | amortizes RTT on fast devices, limits loss on flaky ones |
| Correctness | possibly assumed honest devices | **verification** by re-execution, quarantine, taint repair (Phase 10) | a malicious device returns plausible wrong results |
| Redundancy | ? | **replicas** until the combined reliability meets a target | turns low-reputation devices into usable capacity |
| Normalization | ? | speed term per tile relative to the local time (online) vs makespan / `M_ref` (MILP) | see the observation below |

**Observation (Phase 9 gate, `hetero_faulty6`, α = 0.5).** At job start every remote device has
`rep = 0.5`, so the MILP puts almost all tiles on the trusted local worker (makespan 0.094 s,
objective 1.21). The greedy rule spreads them (makespan 0.023 s) and scores 2.90 on the MILP
objective, a 140 % gap. The online speed term is a per-task finish time relative to `T_loc` and grows
with the queue, whereas the MILP's is the makespan relative to the *whole* local time. At the same α
the greedy therefore weights speed more heavily than the MILP does. This is a design choice to
explore further, not a bug: the greedy cannot know the whole job's makespan when it
decides one task.

## 6. Open questions

1. Does the paper treat reliability per task (as here) or per device/path? Is `exp(−λ·T)` the
   intended time dependence?
2. Is the paper's optimization solved online, or is it a planning model? Is a greedy approximation
   acceptable for the evaluation?
3. How does the paper normalize the speedup term against reliability (our κ)? Is there a
   recommended α?
4. Is redundancy (replication) part of the paper's model, or an extension?
5. Which baselines does the paper compare against (our `local`, `round_robin`, `fastest`,
   `most_reliable`)?
