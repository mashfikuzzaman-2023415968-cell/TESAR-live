# Experiments

Every experiment runs a real coordinator and a real cluster of worker processes on one laptop.
The cluster is described by a profile in `configs/cluster_profiles/`, and the experiment repeats
each parameter combination several times. Results are averaged with **95 % confidence
intervals** (t distribution).

```bash
python experiments/run_matrix.py configs/experiments/e1_baselines.yaml   # runs/e1_baselines/
python experiments/plots.py runs/e1_baselines                            # figures/ + summary.csv
python experiments/run_matrix.py configs/experiments/e2_alpha_sweep.yaml --dry-run   # plan only
python experiments/run_matrix.py configs/experiments/e2_alpha_sweep.yaml --only 3    # first 3 runs
```

* Each run gets a fresh port and a fresh temporary state directory. Run *i* uses seed
  `seed + i` for both the coordinator and the workers, so repetitions differ and remain
  reproducible.
* **Resume:** runs whose `report.json` already exists are skipped. Delete
  `runs/<name>/` to start over.
* `runs/<name>/results.csv` has one row per run: the run's identity and status (`ok`,
  `incorrect` or `failed`), the matrix parameters, and the metrics of spec §11.3. Each run's
  full outputs (`report.json`, `events.jsonl`, `snapshot.json`, images, logs) are in
  `runs/<name>/<point_id>/rep<i>/`.
* Matrix keys are configuration keys (`scheduler.alpha`) or `cluster.<flag>`, which adds a worker
  option to every worker. For example, `cluster.drop_prob: 0.2` adds `--drop-prob 0.2`; the
  profile must have `lab_mode: true`.
* `local_baseline: true` adds one `policy=local` run per repetition. Plots then compute a
  **measured** speedup (local makespan of the same repetition ÷ makespan). Otherwise they use the
  report's estimated speedup (`local_estimate_s / makespan_s`).

## E1 — Baselines (`e1_baselines.yaml`)

* **Question:** how do `local`, `round_robin`, `fastest`, `most_reliable` and `tesar` compare on a
  heterogeneous cluster with unreliable devices (`hetero_faulty6`: flaky phone, dropper,
  crasher)?
* **Hypothesis:** offloading beats `local`. `round_robin` suffers from slow and faulty devices.
  `fastest` is fastest when nothing fails but wastes work on failing devices. `tesar` keeps most of
  the speed with fewer failed attempts.
* **Figures:** `e1_makespan_by_policy` (bars), `e1_failures_by_policy` (timeouts, losses and
  errors per run, stacked).

## E2 — The speedup–reliability tradeoff (`e2_alpha_sweep.yaml`) — headline figure

* **Question:** what does TESAR's α buy? α = 1 weighs only speed, α = 0 only reliability.
* **Hypothesis:** as α grows, makespan falls while failed attempts and wasted work rise. A middle
  α sits near the knee of the curve.
* **Figures:** `e2_tradeoff_curve` (x = α; left axis makespan; right axis wasted work and failed
  attempts) and `e2_pareto` (speedup against the reliability proxy 1 − failed/attempts, one point
  per α). This mirrors the paper's tradeoff.

## E3 — Failure intensity (`e3_failure_sweep.yaml`)

* **Question:** how do policies degrade as every worker drops more results (drop probability 0 →
  0.3, `hetero5`)?
* **Hypothesis:** makespan and wasted work grow with the drop rate. Reliability-aware `tesar`
  (reputation falls, λ rises, replicas) degrades more gracefully than `round_robin` and
  `fastest`.
* **Figure:** `e3_failure_sweep`: makespan and wasted work against drop probability, one line per
  policy.

## E4 — Adaptive chunking (`e4_chunking.yaml`)

* **Question:** does sizing chunks per device (time target, loss tolerance) help on links with
  very different bandwidth (`hetero5` includes a 2 Mbit/s phone)?
* **Hypothesis:** adaptive chunking sends far fewer messages and finishes sooner than fixed
  one-tile chunks.
* **Figure:** `e4_chunking`: makespan and messages sent, adaptive vs fixed, per policy.

## E5 — Malicious workers (`e5_malicious.yaml`)

* **Question:** how much verification does it take to keep the output correct
  (`hetero_malicious6`: one worker lies on 30 % of its results, one on 5 %)?
* **Hypothesis:** the probability that a wrong result reaches the output falls quickly with
  `verification.base_prob`, while overhead grows roughly linearly. Probation already catches
  frequent liars cheaply.
* **Figure:** `e5_malicious`: detection latency, correct-output rate and verification overhead (%)
  against `base_prob`. With low `base_prob` some runs **should** end with incorrect output; that
  is the measurement, not a failure.

## E7 — Greedy vs optimum (`e7_milp_small.yaml`)

* **Question:** how far is the online greedy TESAR rule from the offline optimum of the same model?
* **Method:** each run saves `snapshot.json` (estimates at job start). `plots.py` solves the MILP
  (PuLP/HiGHS) on the first 24 tiles of each snapshot and evaluates the greedy rule statically on
  the same predictions.
* **Figure:** `e7_milp_gap`: box plot of the objective gap (%) per α; values are in `e7_gaps.csv`.
  See `docs/TESAR_ALIGNMENT.md` §5: the two speed terms are normalized differently.

## Threats to validity

* **Simulation ≠ phones.** All workers share one laptop's CPU, so they compete with each other
  and with the coordinator. "Slow" devices are emulated by sleeping (`--slowdown`), not by weaker
  CPUs. Absolute times are much shorter than on real phones.
* **Application-level network shaping ≠ radio.** `--net-delay-ms` / `--net-bw-kbps` shape the
  worker's sends in Python; real WiFi adds contention, retransmissions, power-saving latency and
  asymmetric links (see `OBSERVATIONS.md`).
* **Energy constants are estimates** (spec placeholders until calibrated on phones with
  `mdc.cli calibrate-energy`); energy-related results are qualitative.
* **Short jobs.** Many runs last only seconds, so start-up effects (cold reputations at 0.5,
  benchmark-based first chunks) weigh more than they would in long jobs.
* **Background load.** Other programs on the laptop (browsers) change timings. Run experiments on
  an otherwise idle machine and keep the default repetitions or more.
* **Small n.** With 3–5 repetitions the intervals are wide. Increase `repetitions` before drawing
  firm conclusions.
