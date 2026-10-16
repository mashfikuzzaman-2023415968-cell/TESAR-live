"""Job model: the task DAG, task nodes and attempts (spec 7.2).

An image job is a three-stage DAG with *partial* dependencies::

    stage 1  s1_r00_c00 ... s1_r00_cNN   (one per base tile, no deps)
                 \\    |    /
    stage 2        s2_r00               (row statistics: deps = that row's tiles)
                       \\  ... /
    stage 3              final           (assembly, in-process: deps = all stage 2)

A synthetic job is ``syn_0000 ... syn_NNNN`` -> ``final``.

Task status life cycle::

    PENDING --deps done--> READY --attempt sent--> RUNNING --result--> DONE
                             ^                        |
                             +---- attempt failed ----+
    DONE --(verification pending)--> VERIFYING         (dependents may proceed)
    any done task --invalidate()--> INVALIDATED (re-run) ; its dependents -> PENDING

**Epochs.** Every task carries an ``epoch`` counter, bumped whenever its
result (or an input it depends on) is invalidated. An :class:`Attempt`
records the epochs of its tasks when sent; a result is only accepted if those
epochs are unchanged. So a stage-2 attempt that was computed from a tile later
found to be wrong can never sneak its stale result in.

Every task execution is an Attempt with a unique ``attempt_id``; a task can
have several (retry, replica, verification). The first accepted result wins
and :meth:`Job.mark_done` is idempotent.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.common.protocol import sha256_hex
from mdc.common.timeutil import monotonic
from mdc.tasks.synthetic import xor_digests
from mdc.tasks.tiling import TileGrid

U8Array = npt.NDArray[np.uint8]
FINAL = "final"


class TaskStatus(enum.Enum):
    """Life-cycle state of a TaskNode."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    VERIFYING = "verifying"
    INVALIDATED = "invalidated"
    FAILED = "failed"


# A dependency is satisfied by a result that exists, verified or not
# (optimistic execution; damage is repaired by invalidation).
SATISFIED = frozenset({TaskStatus.DONE, TaskStatus.VERIFYING})
# Tasks the scheduler may (re)send.
RUNNABLE = frozenset({TaskStatus.READY, TaskStatus.INVALIDATED})


@dataclass
class TaskNode:
    """One node of the DAG. Field names are part of the spec (dashboard/experiments)."""

    task_id: str
    stage: int
    kind: str
    deps: list[str]
    status: TaskStatus = TaskStatus.PENDING
    work_units: float = 0.0
    input_bytes_est: int = 0
    output_bytes_est: int = 0
    result: Any = None
    result_sha256: str | None = None
    produced_by: str | None = None
    verified: bool = False
    attempts: list[str] = field(default_factory=list)
    # Not in the spec's list, needed for geometry and stale-result detection:
    row: int | None = None
    col: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0


@dataclass
class Attempt:
    """One execution of a chunk of tasks on one worker (spec 7.2)."""

    attempt_id: str
    task_ids: list[str]
    worker_id: str
    purpose: str = "primary"  # primary | replica | verify | tiebreak
    sent_t: float | None = None
    result_t: float | None = None
    predicted_s: float = 0.0
    timeout_s: float = 0.0
    # inflight|ok|error|timeout|lost|cancelled|late|late_accepted|rejected
    status: str = "inflight"
    bytes_in: int = 0
    bytes_out: int = 0
    compute_ms: float | None = None
    queue_ms: float | None = None
    task_epochs: dict[str, int] = field(default_factory=dict)
    timed_out: bool = False  # stays True even if a late result is accepted afterwards
    predicted_energy_j: float = 0.0  # committed in the energy ledger while in flight
    energy_settled: bool = False


def result_hash(result: Any) -> str:
    """Hash of a task result: SHA-256 of array bytes, or of canonical JSON.

    Used for idempotency checks and (Phase 10) comparing results.
    """
    if isinstance(result, np.ndarray):
        return sha256_hex(np.ascontiguousarray(result).tobytes())
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical.encode("utf-8"))


@dataclass
class Job:
    """A whole job: tasks, attempts, geometry and parameters."""

    job_id: str
    job_type: str  # "image" | "synthetic"
    tasks: dict[str, TaskNode]
    params: dict[str, Any]
    grid: TileGrid | None = None
    padded: U8Array | None = None
    image_path: str | None = None
    attempts: dict[str, Attempt] = field(default_factory=dict)
    policy: str = ""
    policy_params: dict[str, Any] = field(default_factory=dict)
    start_t: float | None = None
    end_t: float | None = None

    def __post_init__(self) -> None:
        self._dependents: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.deps:
                if dep not in self.tasks:
                    raise ValueError(f"task {task.task_id} depends on unknown task {dep}")
                self._dependents[dep].append(task.task_id)
        self._check_acyclic()
        self.refresh()

    # ----------------------------------------------------------- structure

    def _check_acyclic(self) -> None:
        """Kahn's algorithm: every task must be reachable in topological order."""
        indegree = {tid: len(t.deps) for tid, t in self.tasks.items()}
        queue = [tid for tid, deg in indegree.items() if deg == 0]
        seen = 0
        while queue:
            tid = queue.pop()
            seen += 1
            for child in self._dependents[tid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self.tasks):
            raise ValueError("task graph contains a cycle")

    def dependents(self, task_id: str) -> list[str]:
        """Tasks that directly depend on ``task_id``."""
        return list(self._dependents[task_id])

    def task(self, task_id: str) -> TaskNode:
        """Look up a task (KeyError if unknown)."""
        return self.tasks[task_id]

    # ------------------------------------------------------------- readiness

    def deps_satisfied(self, task_id: str) -> bool:
        """All dependencies have a (possibly unverified) result."""
        return all(self.tasks[d].status in SATISFIED for d in self.tasks[task_id].deps)

    def refresh(self) -> list[str]:
        """Promote PENDING tasks whose deps are satisfied to READY; return them."""
        promoted = []
        for task in self.tasks.values():
            if task.status is TaskStatus.PENDING and self.deps_satisfied(task.task_id):
                task.status = TaskStatus.READY
                promoted.append(task.task_id)
        return promoted

    def ready_tasks(self) -> list[TaskNode]:
        """Schedulable tasks in scheduling order.

        Stage 2 before stage 1 (finishing rows keeps the pipeline flowing),
        then by row and column. The stage-3 ``final`` task is never scheduled:
        the coordinator runs it in-process.
        """
        runnable = [
            t
            for t in self.tasks.values()
            if t.status in RUNNABLE and t.stage < 3 and self.deps_satisfied(t.task_id)  # noqa: PLR2004
        ]
        return sorted(runnable, key=lambda t: (-t.stage, t.row or 0, t.col or 0, t.task_id))

    def final_ready(self) -> bool:
        """True when ``final`` can run (all its deps have results)."""
        final = self.tasks.get(FINAL)
        return (
            final is not None
            and final.status in RUNNABLE | {TaskStatus.PENDING}
            and self.deps_satisfied(FINAL)
        )

    # ---------------------------------------------------------- attempts

    def start_attempt(self, attempt: Attempt) -> None:
        """Register an attempt; record task epochs; mark its tasks RUNNING."""
        if attempt.attempt_id in self.attempts:
            raise ValueError(f"duplicate attempt id {attempt.attempt_id}")
        if not attempt.task_ids:
            raise ValueError("an attempt needs at least one task")
        for tid in attempt.task_ids:
            task = self.tasks[tid]
            if task.stage >= 3:  # noqa: PLR2004
                raise ValueError("the final task is never sent to a worker")
            if attempt.purpose == "primary" and task.status not in RUNNABLE | {TaskStatus.RUNNING}:
                raise ValueError(f"task {tid} is {task.status.value}, not runnable")
        attempt.task_epochs = {tid: self.tasks[tid].epoch for tid in attempt.task_ids}
        if attempt.sent_t is None:
            attempt.sent_t = monotonic()
        self.attempts[attempt.attempt_id] = attempt
        for tid in attempt.task_ids:
            task = self.tasks[tid]
            task.attempts.append(attempt.attempt_id)
            if attempt.purpose in {"primary", "replica"} and task.status in RUNNABLE:
                task.status = TaskStatus.RUNNING

    def attempt_is_current(self, attempt: Attempt) -> bool:
        """False if any of its tasks was invalidated since the attempt was sent."""
        return all(self.tasks[tid].epoch == epoch for tid, epoch in attempt.task_epochs.items())

    def inflight_attempts(self, task_id: str) -> list[Attempt]:
        """Attempts for ``task_id`` still waiting for a result."""
        return [
            self.attempts[a]
            for a in self.tasks[task_id].attempts
            if self.attempts[a].status == "inflight"
        ]

    def requeue(self, task_id: str) -> bool:
        """A RUNNING task lost its attempt: make it runnable again.

        Does nothing (returns False) if another attempt for it is still in
        flight or the task already has a result.
        """
        task = self.tasks[task_id]
        if task.status is not TaskStatus.RUNNING or self.inflight_attempts(task_id):
            return False
        task.status = TaskStatus.READY if self.deps_satisfied(task_id) else TaskStatus.PENDING
        return True

    # ------------------------------------------------------------- results

    def mark_done(
        self,
        task_id: str,
        result: Any,
        worker_id: str | None,
        verified: bool = False,
    ) -> bool:
        """Store a task's result. Returns False if it already had one (idempotent).

        Dependents whose deps are now all satisfied become READY.
        """
        task = self.tasks[task_id]
        if task.status in SATISFIED:
            return False
        if task.status is TaskStatus.FAILED:
            return False
        task.result = result
        task.result_sha256 = result_hash(result)
        task.produced_by = worker_id
        task.verified = verified
        task.status = TaskStatus.DONE
        self.refresh()
        return True

    def replace_result(self, task_id: str, result: Any, worker_id: str | None) -> None:
        """Swap in a corrected result (Phase 10) and invalidate everything downstream."""
        task = self.tasks[task_id]
        task.result = result
        task.result_sha256 = result_hash(result)
        task.produced_by = worker_id
        task.verified = True
        task.status = TaskStatus.DONE
        self.invalidate_dependents(task_id)
        self.refresh()

    def invalidate(self, task_id: str) -> list[str]:
        """Throw away ``task_id``'s result so it is re-executed; cascade downstream.

        The task becomes INVALIDATED (runnable again); every task that
        (transitively) depends on it loses its result and becomes PENDING —
        including ``final`` if it already ran. Epochs are bumped so attempts
        already in flight for any of them cannot deliver stale results.
        Returns all affected task ids.
        """
        task = self.tasks[task_id]
        task.result = None
        task.result_sha256 = None
        task.produced_by = None
        task.verified = False
        task.status = TaskStatus.INVALIDATED
        task.epoch += 1
        return [task_id, *self.invalidate_dependents(task_id)]

    def invalidate_dependents(self, task_id: str) -> list[str]:
        """Reset every transitive dependent of ``task_id`` to PENDING (results dropped)."""
        affected: list[str] = []
        stack = list(self._dependents[task_id])
        seen: set[str] = set()
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            dep = self.tasks[tid]
            if dep.status is TaskStatus.PENDING:
                # It never got a result and nothing is in flight for it (it would
                # be RUNNING), so none of *its* dependents can have run either.
                continue
            dep.result = None
            dep.result_sha256 = None
            dep.produced_by = None
            dep.verified = False
            dep.status = TaskStatus.PENDING
            dep.epoch += 1
            affected.append(tid)
            stack.extend(self._dependents[tid])
        return affected

    def is_complete(self) -> bool:
        """True when every task (including ``final``) has a result."""
        return all(t.status in SATISFIED for t in self.tasks.values())

    # ------------------------------------------------------------- inputs

    def stage1_ids(self) -> list[str]:
        """Stage-1 task ids in row-major order."""
        tasks = [t for t in self.tasks.values() if t.stage == 1]
        return [t.task_id for t in sorted(tasks, key=lambda t: (t.row or 0, t.col or 0))]

    def tile_id(self, row: int, col: int) -> str:
        """Task id of a stage-1 tile."""
        return f"s1_r{row:02d}_c{col:02d}"

    def chunk_columns(self, task_ids: list[str]) -> tuple[int, int, int]:
        """``(row, c0, c1)`` of a chunk; checks it is a contiguous run in one row."""
        tasks = [self.tasks[t] for t in task_ids]
        if not tasks or any(t.kind != "img_tile" for t in tasks):
            raise ValueError("a chunk must contain only img_tile tasks")
        rows = {t.row for t in tasks}
        cols = sorted(t.col for t in tasks if t.col is not None)
        if len(rows) != 1 or len(cols) != len(tasks):
            raise ValueError("chunk tiles must lie in one row")
        if cols != list(range(cols[0], cols[0] + len(cols))):
            raise ValueError("chunk tiles must be contiguous")
        (row,) = rows
        if row is None:  # pragma: no cover - img_tile tasks always have a row
            raise ValueError("tile without a row")
        return row, cols[0], cols[-1]

    def task_input(self, task_ids: list[str]) -> tuple[str, dict[str, Any], U8Array | None]:
        """``(kind, params, input array)`` to send for an attempt covering ``task_ids``."""
        first = self.tasks[task_ids[0]]
        if first.kind == "img_tile":
            if self.grid is None or self.padded is None:
                raise ValueError("image job without geometry")
            row, c0, c1 = self.chunk_columns(task_ids)
            return "img_tile", dict(first.params), self.grid.chunk_input(self.padded, row, c0, c1)
        if len(task_ids) != 1:
            raise ValueError(f"{first.kind} tasks are never chunked")
        if first.kind == "row_stats":
            if self.grid is None or first.row is None:
                raise ValueError("row_stats task without geometry")
            return "row_stats", dict(first.params), self.row_strip(first.row)
        return first.kind, dict(first.params), None

    def expected_output_shape(self, task_ids: list[str]) -> tuple[int, int] | None:
        """Output array shape of a chunk (``None`` for JSON-result tasks)."""
        if self.tasks[task_ids[0]].kind != "img_tile" or self.grid is None:
            return None
        row, c0, c1 = self.chunk_columns(task_ids)
        return self.grid.chunk_rect(row, c0, c1).shape

    def tile_results(self) -> dict[tuple[int, int], U8Array]:
        """``{(row, col): output}`` of every stage-1 tile that has a result."""
        out: dict[tuple[int, int], U8Array] = {}
        for t in self.tasks.values():
            if t.kind == "img_tile" and t.result is not None and t.row is not None:
                out[(t.row, t.col or 0)] = t.result
        return out

    def row_strip(self, row: int) -> U8Array:
        """Stage-2 input: the stitched stage-1 outputs of one tile row."""
        if self.grid is None:
            raise ValueError("not an image job")
        return self.grid.row_strip(self.tile_results(), row)

    def apply_chunk_output(
        self, task_ids: list[str], output: U8Array, worker_id: str | None
    ) -> list[str]:
        """Split a chunk's output per tile and mark each tile done; return accepted ids."""
        if self.grid is None:
            raise ValueError("not an image job")
        row, c0, c1 = self.chunk_columns(task_ids)
        tiles = self.grid.split_chunk_output(output, row, c0, c1)
        accepted = []
        for (r, c), tile in tiles.items():
            tid = self.tile_id(r, c)
            if self.mark_done(tid, tile, worker_id):
                accepted.append(tid)
        return accepted

    # -------------------------------------------------------------- final

    def run_final(self) -> dict[str, Any]:
        """Stage 3, in-process: assemble/aggregate and mark ``final`` done.

        Image job: returns ``{"output": image, "edge_pixels", "pixels",
        "edge_density"}``. Synthetic job: ``{"digest": xor of all digests}``.
        """
        if not self.final_ready():
            raise ValueError("final is not ready")
        summary: dict[str, Any]
        if self.job_type == "image":
            if self.grid is None:
                raise ValueError("image job without geometry")
            stats = [self.tasks[d].result for d in self.tasks[FINAL].deps]
            edge = sum(int(s["edge_pixels"]) for s in stats)
            pixels = sum(int(s["pixels"]) for s in stats)
            summary = {
                "output": self.grid.assemble(self.tile_results()),
                "edge_pixels": edge,
                "pixels": pixels,
                "edge_density": edge / pixels if pixels else 0.0,
            }
            self.mark_done(FINAL, {k: v for k, v in summary.items() if k != "output"}, None)
        else:
            digests = [str(self.tasks[d].result["digest"]) for d in self.tasks[FINAL].deps]
            summary = {"digest": xor_digests(digests)}
            self.mark_done(FINAL, summary, None)
        return summary

    def status_counts(self) -> dict[str, int]:
        """Number of tasks per status (for logs and the dashboard)."""
        counts = {s.value: 0 for s in TaskStatus}
        for t in self.tasks.values():
            counts[t.status.value] += 1
        return counts


def execute_inprocess(job: Job, chunk_sizes: list[int] | None = None) -> dict[str, Any]:
    """Run a whole job in this process through the task registry; return ``run_final()``.

    Uses the same steps as a distributed run — attempts, ``task_input``,
    ``mdc.tasks.base.execute``, ``apply_chunk_output``/``mark_done`` — just
    without the network. Stage-1 tiles are grouped into chunks whose sizes
    cycle through ``chunk_sizes`` (default: one tile per chunk). Used by
    tests, the phase gate and the CLI.
    """
    from mdc.common.ids import new_id  # noqa: PLC0415 - avoid import cost for library users
    from mdc.tasks.base import execute  # noqa: PLC0415

    sizes = chunk_sizes or [1]
    turn = 0
    while not job.final_ready():
        ready = job.ready_tasks()
        if not ready:
            raise RuntimeError("job stalled: no ready tasks and final not ready")
        task = ready[0]
        task_ids = [task.task_id]
        if task.kind == "img_tile" and job.grid is not None and task.row is not None:
            want = sizes[turn % len(sizes)]
            turn += 1
            col = task.col or 0
            while len(task_ids) < want and col + len(task_ids) < job.grid.cols:
                nxt = job.tasks[job.tile_id(task.row, col + len(task_ids))]
                if nxt.status not in RUNNABLE:
                    break
                task_ids.append(nxt.task_id)
        attempt = Attempt(attempt_id=new_id("att"), task_ids=task_ids, worker_id="inprocess")
        job.start_attempt(attempt)
        kind, params, data = job.task_input(task_ids)
        output = execute(kind, params, data)
        attempt.status = "ok"
        if output.array is not None:
            job.apply_chunk_output(task_ids, output.array, "inprocess")
        else:
            job.mark_done(task_ids[0], output.json_result, "inprocess")
    return job.run_final()
