"""Task-kind registry and cost model (spec 8.6, S.7.1).

``TASK_KINDS`` maps each kind name to its :class:`TaskKind`. This dictionary
*is* the list of things a worker can be asked to do: the worker looks the kind
up here and calls ``execute``. There is no other path to code execution —
which is how "code offloading" is done securely (no code over the network).

Every ``execute``:

1. validates ``params`` against the hard limits (``validate_task_params``) and
   the input array against the kind (``validate_task_input``) — the worker
   enforces its own copy of the limits, never trusting the coordinator;
2. runs the kernel with a cooperative ``deadline``;
3. returns a :class:`TaskOutput` with either an array or a JSON result,
   validated before returning (a bug cannot emit a malformed result).

Cost model (``work_units``), roughly proportional to compute time:

* ``img_tile``:  ``(core_h * core_w / 1e6) * iters * (2r + 1)``
* ``row_stats``: ``pixels / 1e7``
* ``synthetic``: ``chain_len / 1000``

Predicted compute time on device d: ``work_units * ms_per_unit[d][kind] / 1000``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mdc.common import validation as V
from mdc.common.validation import DEFAULT_TASK_LIMITS, TaskLimits, ValidationError
from mdc.tasks.image_pipeline import halo as pipeline_halo
from mdc.tasks.image_pipeline import pipeline_valid
from mdc.tasks.row_stats import row_stats
from mdc.tasks.synthetic import hash_chain

ArrayOrNone = npt.NDArray[Any] | None


@dataclass(frozen=True)
class TaskOutput:
    """Result of executing one task: exactly one of ``array`` / ``json_result``."""

    array: npt.NDArray[np.uint8] | None = None
    json_result: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskKind:
    """A registered computation."""

    name: str
    execute: Callable[[dict[str, Any], ArrayOrNone, TaskLimits, float | None], TaskOutput]
    needs_input: bool


def _validated(kind: str, params: object, data: ArrayOrNone, limits: TaskLimits) -> dict[str, Any]:
    checked = V.validate_task_params(kind, params, limits)
    if data is not None and not isinstance(data, np.ndarray):
        raise ValidationError("task input must be a numpy array")
    shape = None if data is None else tuple(int(d) for d in data.shape)
    dtype = None if data is None else data.dtype.name
    V.validate_task_input(kind, checked, shape, dtype, limits)
    return checked


def _exec_img_tile(
    params: dict[str, Any], data: ArrayOrNone, limits: TaskLimits, deadline: float | None
) -> TaskOutput:
    p = _validated("img_tile", params, data, limits)
    if data is None:  # pragma: no cover - rejected by validate_task_input
        raise ValidationError("img_tile requires an input array")
    out = pipeline_valid(data, p["r"], p["iters"], limits, deadline)
    h = pipeline_halo(p["r"], p["iters"])
    V.validate_result_shape((data.shape[0] - 2 * h, data.shape[1] - 2 * h), out.shape)
    return TaskOutput(array=out)


def _exec_row_stats(
    params: dict[str, Any], data: ArrayOrNone, limits: TaskLimits, deadline: float | None
) -> TaskOutput:
    p = _validated("row_stats", params, data, limits)
    if data is None:  # pragma: no cover - rejected by validate_task_input
        raise ValidationError("row_stats requires an input array")
    result = row_stats(data, p["edge_threshold"])
    return TaskOutput(json_result=V.validate_json_result("row_stats", result))


def _exec_synthetic(
    params: dict[str, Any], data: ArrayOrNone, limits: TaskLimits, deadline: float | None
) -> TaskOutput:
    p = _validated("synthetic", params, data, limits)
    digest = hash_chain(p["seed"], p["chain_len"], limits, deadline)
    return TaskOutput(json_result=V.validate_json_result("synthetic", {"digest": digest}))


TASK_KINDS: dict[str, TaskKind] = {
    "img_tile": TaskKind("img_tile", _exec_img_tile, needs_input=True),
    "row_stats": TaskKind("row_stats", _exec_row_stats, needs_input=True),
    "synthetic": TaskKind("synthetic", _exec_synthetic, needs_input=False),
}


def execute(
    kind: str,
    params: object,
    data: ArrayOrNone = None,
    limits: TaskLimits = DEFAULT_TASK_LIMITS,
    deadline: float | None = None,
) -> TaskOutput:
    """Run a registered task kind (the single entry point used by workers)."""
    task_kind = TASK_KINDS.get(kind)
    if task_kind is None:
        raise ValidationError(f"unknown task kind {kind!r}")
    if not isinstance(params, dict):
        raise ValidationError("params must be an object")
    return task_kind.execute(params, data, limits, deadline)


# ------------------------------------------------------------- cost model


def img_tile_work_units(core_h: int, core_w: int, r: int, iters: int) -> float:
    """Work units of an image chunk with an output (core) of ``core_h x core_w``."""
    return (core_h * core_w / 1e6) * iters * (2 * r + 1)


def row_stats_work_units(pixels: int) -> float:
    """Work units of a row-statistics task."""
    return pixels / 1e7


def synthetic_work_units(chain_len: int) -> float:
    """Work units of a hash-chain task."""
    return chain_len / 1000


def compute_seconds(work_units: float, ms_per_unit: float) -> float:
    """Predicted compute time: ``work_units * ms_per_unit / 1000``."""
    return work_units * ms_per_unit / 1000.0
