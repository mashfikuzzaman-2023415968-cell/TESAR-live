"""Adaptive chunk sizing (spec Phase 6 item 3, new feature A).

A chunk is a run of contiguous base tiles in one row sent as one attempt.
The size is a tradeoff:

* **bigger chunks** amortize the per-message costs — one round trip, one
  header, one shared halo instead of one per tile — which matters most on
  high-latency links;
* **smaller chunks** limit the work lost when a device fails mid-chunk: an
  unreliable phone should never hold a big piece of the job.

Rule (per worker ``d``, starting at tile ``first``)::

    per_tile_s = compute_s(one tile, d) + (tile_in + tile_out) / bw(d)
    n_time     = floor((target_task_s - rtt(d)) / per_tile_s)
                 # aim for attempts of about target_task_s seconds
    n_loss     = floor(-ln(1 - loss_tolerance) / (lambda(d) * per_tile_s))
                 # keep P(chunk lost) = 1 - exp(-lambda * n * per_tile_s) <= tolerance
                 # (infinite when lambda == 0)
    n          = clamp(min(n_time, n_loss), min_chunk_tiles, max_chunk_tiles)

and finally ``n`` is cut to the number of contiguous runnable tiles left in
the row (done by ``SchedulingContext.contiguous_ready``).

With ``chunking.adaptive: false`` every worker gets ``fixed_chunk_tiles``.
"""

from __future__ import annotations

import math

from mdc.coordinator.job import TaskNode
from mdc.coordinator.policies.base import SchedulingContext, WorkerView

UNBOUNDED = 10**9


def choose_chunk_tiles(worker: WorkerView, first: TaskNode, ctx: SchedulingContext) -> int:
    """How many base tiles worker ``worker`` should get, starting at ``first``.

    The result is *before* the row-boundary cut; use
    :func:`chunk_for` to get the actual task ids.
    """
    cfg = ctx.config.chunking
    if not cfg.adaptive:
        return cfg.fixed_chunk_tiles
    if first.kind != "img_tile":
        return 1
    one = (first.task_id,)
    tile_in, tile_out = ctx.wire_bytes(one)
    per_tile_s = ctx.compute_s(one, worker) + (tile_in + tile_out) / max(1.0, worker.bw_bytes_per_s)
    if per_tile_s <= 0:
        return cfg.max_chunk_tiles
    n_time = math.floor((cfg.target_task_s - worker.rtt_s) / per_tile_s)
    lam = worker.failure_rate_per_s
    if lam > 0:
        n_loss = math.floor(-math.log(1.0 - cfg.loss_tolerance) / (lam * per_tile_s))
    else:
        n_loss = UNBOUNDED
    return max(cfg.min_chunk_tiles, min(cfg.max_chunk_tiles, n_time, n_loss))


def chunk_for(
    worker: WorkerView, first: TaskNode, ctx: SchedulingContext, taken: set[str]
) -> list[str]:
    """Task ids of the chunk for ``worker`` starting at ``first`` (row-boundary aware)."""
    return ctx.contiguous_ready(first, choose_chunk_tiles(worker, first, ctx), taken)
