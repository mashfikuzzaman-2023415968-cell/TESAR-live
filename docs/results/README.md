# First experiment results (laptop-only, 2026-10-16)

All six experiments were run once in full on the development laptop: 145 runs in about 12
minutes, every run with status `ok`. Figures and `summary.csv` (mean, 95 % CI half-width, n per
point) are copied here from `runs/<experiment>/figures/`. Re-create them with
`python experiments/run_matrix.py configs/experiments/<e>.yaml && python experiments/plots.py runs/<e>`.
Absolute numbers depend on this machine and its load; see the threats to validity in
`docs/EXPERIMENTS.md`.

| Exp. | What the data shows (mean makespan, 95 % CI) | Reading |
|---|---|---|
| E1 | local 0.25 ± 0.12 s; tesar 0.49 ± 0.04 s; fastest 1.47 ± 1.99 s; most_reliable 2.02 ± 2.02 s; round_robin 4.48 ± 2.02 s | **On one laptop, `local` wins**: every emulated "remote" worker shares the same CPU and adds TLS, serialization and verification, so offloading cannot pay off for sub-second jobs. Among offloading policies, `tesar` is fastest and by far the most *predictable* (narrow CI): replicas absorb the flaky and crashing devices that occasionally stall `fastest` and `most_reliable`. |
| E2 | α = 0: 1.08 ± 1.70 s; α = 0.25…1: 0.49–0.54 s | Pure reliability (α = 0) waits for the trusted local worker and is slow and erratic; any α ≥ 0.25 is fast. With 20 replicas per run the reliability term is mostly satisfied by replication, so the curve is flat above 0.25: the tradeoff is visible only at the reliability end on this setup. |
| E3 | drop 0 → 0.3: round_robin 0.20 → 10.5 s; fastest 0.23 → 2.97 s; tesar 0.46 → 2.52 s | Failures hurt `round_robin` most (it keeps feeding droppers). `tesar` starts slower (replication cost) and degrades least, ending fastest at 30 % drops. |
| E4 | fastest: adaptive 0.70 vs fixed 0.74 s; tesar: adaptive 1.12 vs fixed 0.76 s | Adaptive chunking sends far fewer messages, but with `tesar` big chunks are replicated as a whole, so it was slower here: chunking and replication interact (bigger chunk → lower R → more replicated work). |
| E5 | correct-output rate 1.0 except base_prob 0.05: 0.8; overhead 14 % → 20 % from base_prob 0 to 1 | Probation alone catches the 30 % liar; the 5 % liar occasionally slips a wrong tile through at a low sampling rate, which is the residual risk in SECURITY.md. Overhead grows slowly because replicas already provide free comparisons. |
| E7 | gap greedy vs MILP: α 0.25 ≈ 190–280 %, 0.5 ≈ 140–170 %, 0.75 ≈ 0–25 % | The online rule is near-optimal when speed dominates, and far from the model's optimum when reliability matters: the MILP keeps cold-start (rep 0.5) work on the trusted local worker. See `docs/TESAR_ALIGNMENT.md` §5. |

These are single-laptop simulations with 3–5 repetitions. They show that the system and its
measurements work, and where to look in the real-phone experiments. They are not final research
conclusions.
