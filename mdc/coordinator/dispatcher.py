"""Sending tasks and receiving results (spec Phase 4 item 7, S.8).

Send path (:meth:`Dispatcher.send`):

1. take the chunk's input from the job (``job.task_input``), pack it,
2. create an :class:`Attempt` (unique ``attempt_id``, ``predicted_s``,
   ``timeout_s``), register it in the job (tasks become RUNNING) and in the
   worker's ``inflight`` map, remember which *session* it was sent to,
3. write the frame in a background task (so a slow link never blocks the
   scheduler); if the write fails the attempt is ``lost`` and requeued.

Result path (:meth:`Dispatcher.on_result`) — every check happens **before**
anything is accepted:

* unknown ``attempt_id`` → *unsolicited*; more than 3 per minute from one
  connection → disconnect and strike (S.8.3);
* attempt sent to another session → ``result_from_wrong_worker``, rejected,
  counted as a failure of the sender (S.8.2 — a worker cannot submit results
  for someone else's work);
* attempt no longer in flight → ``late``, ignored (Phase 5 accepts some);
* ``status: error`` → attempt ``error``, tasks back to READY;
* payload fails its SHA-256 → *transmission* error (not dishonesty): ``error``,
  rescheduled;
* wrong output shape / malformed JSON result → ``rejected``, security-logged;
* stale (an input changed since sending — see ``Job.attempt_is_current``) →
  discarded.

Accepted results are split per tile, stored with ``job.mark_done`` (which is
idempotent: the first accepted result wins), the worker's speed estimate is
refined, other in-flight copies of the same tasks are cancelled, and the
scheduler is woken up.

Worker-reported ``compute_ms``/``queue_ms`` are clamped to what the
coordinator measured itself (S.6.8).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mdc.common import protocol as P
from mdc.common.config import Config
from mdc.common.events import EventLogger
from mdc.common.ids import new_id
from mdc.common.ratelimit import SlidingCounter
from mdc.common.timeutil import monotonic
from mdc.common.validation import ValidationError, clamp_durations, validate_json_result
from mdc.coordinator.estimators import Estimators
from mdc.coordinator.job import SATISFIED, Attempt, Job
from mdc.coordinator.policies.base import Assignment, chunk_compute_s, predict_seconds
from mdc.coordinator.registry import Registry, WorkerRecord, WorkerState
from mdc.coordinator.reputation import battery_eligible
from mdc.coordinator.security_log import SecurityLog
from mdc.coordinator.verifier import PRODUCING, Verifier

log = logging.getLogger(__name__)
CHECKING = frozenset({"verify", "tiebreak"})  # attempts that judge results, never store them

UNSOLICITED_PER_MINUTE = 3


class Channel(Protocol):
    """The part of a server connection the dispatcher uses."""

    session_id: str
    worker_id: str
    unsolicited: SlidingCounter

    async def send(self, msg: P.Message) -> int:
        """Write one frame; return bytes written."""
        ...

    def close(self) -> None:
        """Drop the connection."""
        ...


@dataclass
class TrafficCounters:
    """Message/byte counters for the run report."""

    messages_sent: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0


class Dispatcher:
    """Turns assignments into TASK_ASSIGN frames and TASK_RESULT frames into progress."""

    def __init__(
        self,
        config: Config,
        registry: Registry,
        events: EventLogger,
        security_log: SecurityLog,
        channel_for: Callable[[str], Channel | None],
        on_progress: Callable[[], None],
        estimators: Estimators | None = None,
    ) -> None:
        self.cfg = config
        self.registry = registry
        self.events = events
        self.security_log = security_log
        self.channel_for = channel_for
        self.on_progress = on_progress
        self.estimators = estimators if estimators is not None else Estimators.from_config(config)
        self.reputation = self.estimators.reputation
        self.energy = self.estimators.energy
        self.netest = self.estimators.netest
        self.job: Job | None = None
        self.traffic = TrafficCounters()
        self._attempt_session: dict[str, str] = {}
        self._send_tasks: set[asyncio.Task[None]] = set()
        self.verifier = Verifier(config, self)

    def start_job(self, job: Job) -> None:
        """Attach the job whose tasks are dispatched from now on."""
        self.job = job
        self._attempt_session.clear()
        self.verifier.start_job()

    # ------------------------------------------------------------ predictions

    def queue_wait_s(self, record: WorkerRecord, now: float | None = None) -> float:
        """Remaining predicted time of ``record``'s in-flight attempts / its slots."""
        now = monotonic() if now is None else now
        remaining = sum(
            max(0.0, a.predicted_s - (now - (a.sent_t or now))) for a in record.inflight.values()
        )
        return remaining / max(1, record.slots)

    def predict_s(self, task_ids: list[str] | tuple[str, ...], record: WorkerRecord) -> float:
        """``queue_wait + rtt + wire_bytes / bw + compute`` (spec Phase 6 item 2).

        Used for assignments that arrive without a prediction (``round_robin``,
        ``local``, fallback); ``fastest`` and later policies supply their own,
        computed the same way.
        """
        if self.job is None:
            return 0.0
        return predict_seconds(
            self.job,
            task_ids,
            record.ms_per_unit,
            self.netest.rtt(record.worker_id),
            self.netest.bw(record.worker_id),
            self.queue_wait_s(record),
            self.netest.compression_table(),
        )

    def predict_r(self, record: WorkerRecord, predicted_s: float) -> float:
        """``R = rep * exp(-λ * predicted_s)`` for an attempt on ``record``."""
        return self.reputation.reliability(
            record.worker_id,
            predicted_s,
            trusted=record.trusted,
            eligible=battery_eligible(record.battery, self.cfg.workers.min_battery_pct),
        )

    def timeout_s(self, predicted_s: float) -> float:
        """``max(min, factor * predicted + slack)`` (spec Phase 5 item 2)."""
        lv = self.cfg.liveness
        return max(
            lv.task_timeout_min_s, lv.task_timeout_factor * predicted_s + lv.task_timeout_slack_s
        )

    # ------------------------------------------------------------------ send

    def send(self, assignment: Assignment) -> Attempt | None:
        """Create and send one attempt. Returns ``None`` if it could not be sent."""
        job = self.job
        record = self.registry.get(assignment.worker_id)
        channel = self.channel_for(assignment.worker_id)
        if job is None or record is None or channel is None:
            return None
        task_ids = list(assignment.task_ids)
        kind, params, data = job.task_input(task_ids)
        fields: dict[str, Any] = {}
        payload = b""
        if data is not None:
            fields, payload = P.pack_array(
                np.ascontiguousarray(data),
                self.cfg.network.compression,
                self.cfg.network.zlib_level,
            )
        predicted = (
            assignment.predicted_s
            if assignment.predicted_s is not None
            else self.predict_s(task_ids, record)
        )
        if data is not None:
            self.netest.on_compression(kind, "in", int(data.nbytes), len(payload))
        compute_s = chunk_compute_s(job, task_ids, record.ms_per_unit)
        transfer_s = max(0.0, predicted - compute_s - self.queue_wait_s(record))
        energy_j = self.energy.task_energy_j(record.device_class, compute_s, transfer_s)
        attempt = Attempt(
            attempt_id=new_id("att"),
            task_ids=task_ids,
            worker_id=record.worker_id,
            purpose=assignment.purpose,
            predicted_s=predicted,
            timeout_s=self.timeout_s(predicted),
            bytes_in=len(payload),
            predicted_energy_j=energy_j,
        )
        job.start_attempt(attempt)
        self.energy.commit(record.worker_id, energy_j)
        record.inflight[attempt.attempt_id] = attempt
        record.attempts_sent += 1
        record.chunk_tiles_sent += len(task_ids)
        self._attempt_session[attempt.attempt_id] = channel.session_id
        msg = P.make_message(
            P.TASK_ASSIGN,
            payload,
            task_id=task_ids[0],
            attempt_id=attempt.attempt_id,
            kind=kind,
            params=params,
            **fields,
        )
        self.events.emit(
            "attempt_sent",
            attempt_id=attempt.attempt_id,
            task_ids=task_ids,
            n_tiles=len(task_ids),
            worker_id=record.worker_id,
            purpose=attempt.purpose,
            predicted_s=round(predicted, 4),
            timeout_s=round(attempt.timeout_s, 4),
            bytes_in=len(payload),
            predicted_R=round(self.predict_r(record, predicted), 6),
            predicted_energy_j=round(energy_j, 4),
            est_rtt_s=round(self.netest.rtt(record.worker_id), 5),
            est_bw_bytes_per_s=round(self.netest.bw(record.worker_id), 1),
        )
        task = asyncio.create_task(self._write(channel, msg, attempt))
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)
        return attempt

    async def _write(self, channel: Channel, msg: P.Message, attempt: Attempt) -> None:
        try:
            sent = await channel.send(msg)
        except (OSError, P.WireError, ConnectionError) as exc:
            log.info("sending %s failed: %s", attempt.attempt_id, type(exc).__name__)
            self.mark_lost(attempt, "send_failed")
            self.on_progress()
            return
        self.traffic.messages_sent += 1
        self.traffic.bytes_sent += sent

    # -------------------------------------------------------------- failures

    def _finish(self, attempt: Attempt, status: str) -> WorkerRecord | None:
        """Close an attempt: status, inflight bookkeeping, counters."""
        attempt.status = status
        if attempt.result_t is None:
            attempt.result_t = monotonic()
        record = self.registry.get(attempt.worker_id)
        if record is not None:
            record.inflight.pop(attempt.attempt_id, None)
        if attempt.purpose in CHECKING and status != "ok":
            self.verifier.on_verify_failed(attempt)  # requeue the check
        if not attempt.energy_settled:
            attempt.energy_settled = True
            actual = self._measured_energy_j(attempt, record) if status == "ok" else None
            self.energy.settle(attempt.worker_id, attempt.predicted_energy_j, actual)
            if record is not None:
                record.energy_spent_j = self.energy.ledger(record.worker_id).spent_j
        return record

    def _measured_energy_j(self, attempt: Attempt, record: WorkerRecord | None) -> float | None:
        """Energy of a finished attempt from its measured times (spec Phase 8 item 2):
        the worker's ``compute_ms`` (already clamped to the measured total) and the
        rest of the coordinator-measured round trip as transfer."""
        if record is None or attempt.compute_ms is None or attempt.sent_t is None:
            return None
        total_s = (attempt.result_t or attempt.sent_t) - attempt.sent_t
        compute_s = attempt.compute_ms / 1000.0
        transfer_s = max(0.0, total_s - compute_s - (attempt.queue_ms or 0.0) / 1000.0)
        return self.energy.task_energy_j(record.device_class, compute_s, transfer_s)

    def _requeue(self, attempt: Attempt) -> None:
        if self.job is None:
            return
        for tid in attempt.task_ids:
            self.job.requeue(tid)

    def mark_lost(self, attempt: Attempt, reason: str) -> None:
        """The worker went away (or the send failed): requeue the attempt's tasks."""
        if attempt.status != "inflight":
            return
        record = self._finish(attempt, "lost")
        if record is not None:
            record.failed += 1
        self.reputation.record_failure(attempt.worker_id, "disconnect")
        self.reputation.record_attempt(attempt.worker_id, True, attempt.predicted_s)
        self.events.emit(
            "attempt_lost",
            attempt_id=attempt.attempt_id,
            worker_id=attempt.worker_id,
            reason=reason,
        )
        self._requeue(attempt)

    def on_disconnect(self, record: WorkerRecord, reason: str) -> int:
        """Mark every in-flight attempt of ``record`` lost; return how many."""
        attempts = list(record.inflight.values())
        for attempt in attempts:
            self.mark_lost(attempt, reason)
        if attempts:
            self.on_progress()
        return len(attempts)

    def _send_cancel(self, attempt: Attempt) -> None:
        """Tell the worker to drop an attempt (only on the session it was sent to)."""
        channel = self.channel_for(attempt.worker_id)
        if channel is None or self._attempt_session.get(attempt.attempt_id) != (channel.session_id):
            return
        task = asyncio.create_task(
            self._quiet_send(channel, P.make_message(P.TASK_CANCEL, attempt_id=attempt.attempt_id))
        )
        self._send_tasks.add(task)
        task.add_done_callback(self._send_tasks.discard)

    def cancel(self, attempt: Attempt, reason: str) -> None:
        """Cancel an in-flight attempt (its result is no longer needed)."""
        if attempt.status != "inflight":
            return
        self._finish(attempt, "cancelled")
        self._send_cancel(attempt)
        log.debug("cancelled %s (%s)", attempt.attempt_id, reason)

    def sweep_timeouts(self, now: float) -> int:
        """Time out every in-flight attempt older than its ``timeout_s`` (spec Phase 5 item 2).

        A timed-out attempt is cancelled on the worker, counted as a ``timeout``
        failure, and its tasks go back to READY — unless another attempt for the
        same task is still in flight, in which case we simply wait for that one
        (``Job.requeue`` checks). Called every 0.25 s by the failure detector.
        """
        expired = [
            attempt
            for record in self.registry
            for attempt in list(record.inflight.values())
            if attempt.sent_t is not None and now - attempt.sent_t > attempt.timeout_s
        ]
        for attempt in expired:
            waited = now - (attempt.sent_t or now)
            attempt.timed_out = True
            record = self._finish(attempt, "timeout")
            if record is not None:
                record.failed += 1
            self.reputation.record_failure(attempt.worker_id, "timeout")
            self.reputation.record_attempt(attempt.worker_id, True, attempt.predicted_s)
            self.events.emit(
                "attempt_timeout",
                attempt_id=attempt.attempt_id,
                worker_id=attempt.worker_id,
                waited_s=round(waited, 3),
            )
            self._send_cancel(attempt)
            self._requeue(attempt)
        if expired:
            self.on_progress()
        return len(expired)

    async def _quiet_send(self, channel: Channel, msg: P.Message) -> None:
        with contextlib.suppress(OSError, P.WireError, ConnectionError):
            self.traffic.bytes_sent += await channel.send(msg)
            self.traffic.messages_sent += 1

    # --------------------------------------------------------------- results

    def on_result(self, channel: Channel, msg: P.Message) -> None:
        """Handle one (schema-validated) TASK_RESULT from ``channel``.

        Three steps, each of which may reject the result: who sent it
        (:meth:`_owned_attempt`), is it still wanted and well-formed
        (:meth:`_decode`), then acceptance (:meth:`_accept`).
        """
        self.traffic.bytes_received += len(msg.payload)
        job = self.job
        attempt = self._owned_attempt(channel, msg.header["attempt_id"])
        if job is None or attempt is None:
            return
        sender = self.registry.get(channel.worker_id)
        if sender is not None and sender.state is WorkerState.QUARANTINED:
            self.security_log.emit(
                "result_from_quarantined",
                worker_id=channel.worker_id,
                attempt_id=attempt.attempt_id,
            )
            return
        late = attempt.status != "inflight"
        if late and self._duplicate_wanted(attempt, msg.header["status"]):
            output = self._decode(job, attempt, msg, late)
            if output is not None:
                self.verifier.on_duplicate(attempt, output)
                self.on_progress()
            return
        if late and not self._late_result_wanted(job, attempt, msg.header["status"]):
            self._emit_late(attempt, accepted=False)
            return
        output = self._decode(job, attempt, msg, late)
        if output is None:
            return
        if attempt.purpose in CHECKING:
            self._verified(job, attempt, output)
            return
        self._accept(job, attempt, output, late)

    def _duplicate_wanted(self, attempt: Attempt, status: str) -> bool:
        """A cancelled copy (replica) that finished anyway: compare it for free."""
        return (
            self.verifier.cfg.enabled
            and attempt.status == "cancelled"
            and attempt.purpose in PRODUCING
            and status == "ok"
        )

    def _verified(self, job: Job, attempt: Attempt, output: Any) -> None:
        """A verification/tiebreak result: the verifier judges it; nothing is stored."""
        self._finish(attempt, "ok")
        total_s = (attempt.result_t or 0.0) - (attempt.sent_t or 0.0)
        record = self.registry.get(attempt.worker_id)
        if record is not None:
            record.note_success()
            work = sum(job.tasks[t].work_units for t in attempt.task_ids)
            record.observe_speed(
                job.tasks[attempt.task_ids[0]].kind, attempt.compute_ms or 0.0, work
            )
        self.reputation.record_success(attempt.worker_id)
        self.reputation.record_attempt(attempt.worker_id, False, attempt.predicted_s)
        self._emit_result(attempt, "ok", total_s)
        self.verifier.on_verify_result(attempt, output)
        self.on_progress()

    def _owned_attempt(self, channel: Channel, attempt_id: str) -> Attempt | None:
        """The attempt if ``channel`` may report it; otherwise handle the abuse, return None.

        Unknown ids are *unsolicited* (rate-limited, S.8.3); an attempt sent to a
        different worker or session is a ``result_from_wrong_worker`` (S.8.2).
        """
        job = self.job
        attempt = job.attempts.get(attempt_id) if job is not None else None
        if attempt is None:
            self._unsolicited(channel, attempt_id)
            return None
        sent_session = self._attempt_session.get(attempt.attempt_id)
        if attempt.worker_id != channel.worker_id or sent_session != channel.session_id:
            self.security_log.emit(
                "result_from_wrong_worker",
                attempt_id=attempt.attempt_id,
                sender=channel.worker_id,
                owner=attempt.worker_id,
            )
            sender = self.registry.get(channel.worker_id)
            if sender is not None:
                sender.failed += 1
            self.reputation.record_failure(channel.worker_id, "error")
            return None
        return attempt

    def _reject(self, attempt: Attempt, late: bool, status: str, detail: str) -> None:
        """A bad result: a late one is just ignored, a current one fails the attempt."""
        if late:
            self._emit_late(attempt, accepted=False)
        else:
            self._fail(attempt, status, detail=detail)

    def _decode(self, job: Job, attempt: Attempt, msg: P.Message, late: bool) -> Any:
        """Record timings, then validate and decode the output; ``None`` if rejected."""
        h = msg.header
        now = monotonic()
        attempt.result_t = now
        attempt.bytes_out = len(msg.payload)
        total_s = now - (attempt.sent_t or now)
        attempt.compute_ms, attempt.queue_ms = clamp_durations(
            h["compute_ms"], h["queue_ms"], total_s
        )
        if h["status"] == "error":
            self._fail(attempt, "error", detail=str(h.get("error")))
            return None
        try:
            output = self._extract_output(job, attempt, msg)
        except P.IntegrityError:
            self._reject(attempt, late, "error", "integrity")
            return None
        except ValidationError as exc:
            self.security_log.emit(
                "validation_failed",
                worker_id=attempt.worker_id,
                attempt_id=attempt.attempt_id,
                detail=str(exc),
            )
            self._reject(attempt, late, "rejected", str(exc))
            return None
        if not job.attempt_is_current(attempt):
            # An input of this chunk was invalidated after sending: stale result.
            if late:
                self._emit_late(attempt, accepted=False)
            else:
                self._finish(attempt, "late")
                self._emit_result(attempt, "late", total_s)
                self.on_progress()
            return None
        return output

    def _accept(self, job: Job, attempt: Attempt, output: Any, late: bool) -> None:
        """Store the result (first result wins), update estimates, cancel duplicates."""
        if isinstance(output, np.ndarray):
            accepted = job.apply_chunk_output(attempt.task_ids, output, attempt.worker_id)
        else:
            first = attempt.task_ids[0]
            accepted = [first] if job.mark_done(first, output, attempt.worker_id) else []
        status = "late_accepted" if late else "ok"
        self._finish(attempt, status)
        if late:
            self._emit_late(attempt, accepted=True)
        record = self.registry.get(attempt.worker_id)
        if record is not None:
            record.completed += 1
            record.tiles_done += len(accepted)
            record.note_success()
            work = sum(job.tasks[t].work_units for t in attempt.task_ids)
            record.observe_speed(
                job.tasks[attempt.task_ids[0]].kind, attempt.compute_ms or 0.0, work
            )
        self.reputation.record_success(attempt.worker_id)
        if not late:  # a late result was already counted as a failed attempt
            self.reputation.record_attempt(attempt.worker_id, False, attempt.predicted_s)
        total_s = (attempt.result_t or 0.0) - (attempt.sent_t or 0.0)
        kind = job.tasks[attempt.task_ids[0]].kind
        if isinstance(output, np.ndarray):
            self.netest.on_compression(kind, "out", int(output.nbytes), attempt.bytes_out)
        self.netest.on_transfer(
            attempt.worker_id,
            total_s,
            attempt.compute_ms,
            attempt.queue_ms,
            attempt.bytes_in,
            attempt.bytes_out,
        )
        self._emit_result(attempt, status, total_s)
        self.verifier.on_accepted(attempt, accepted)
        for tid in accepted:
            self.events.emit(
                "task_done",
                task_id=tid,
                worker_id=attempt.worker_id,
                verified=job.tasks[tid].verified,
            )
        for tid in attempt.task_ids:
            for other in job.inflight_attempts(tid):
                self.cancel(other, "task already done")
        self.on_progress()

    def _extract_output(self, job: Job, attempt: Attempt, msg: P.Message) -> Any:
        """Validate and decode a successful result (raises on anything wrong)."""
        h = msg.header
        first = job.tasks[attempt.task_ids[0]]
        expected_shape = job.expected_output_shape(attempt.task_ids)
        if expected_shape is not None:
            if "shape" not in h:
                raise ValidationError("image result without an array")
            if h["dtype"] != "uint8" or tuple(h["shape"]) != expected_shape:
                raise ValidationError(
                    f"result shape {h['shape']} ({h['dtype']}) != expected {expected_shape}"
                )
            return P.unpack_array(h, msg.payload, self.cfg.network.max_raw_array_bytes)
        if "shape" in h:
            raise ValidationError(f"{first.kind} result must be JSON, not an array")
        result = validate_json_result(first.kind, h.get("json_result"))
        if first.kind == "row_stats" and job.grid is not None and first.row is not None:
            expected_pixels = job.grid.chunk_rect(first.row, 0, job.grid.cols - 1).pixels
            if result["pixels"] != expected_pixels:
                raise ValidationError("row_stats pixel count does not match the row")
        return result

    @staticmethod
    def _late_result_wanted(job: Job, attempt: Attempt, status: str) -> bool:
        """A result after timeout/loss is still useful if one of its tasks lacks a result
        (spec Phase 5 item 4). Error reports that arrive late are never useful."""
        return (
            attempt.status in {"timeout", "lost"}
            and status == "ok"
            and any(job.tasks[t].status not in SATISFIED for t in attempt.task_ids)
        )

    def _emit_late(self, attempt: Attempt, accepted: bool) -> None:
        self.events.emit(
            "attempt_late",
            attempt_id=attempt.attempt_id,
            worker_id=attempt.worker_id,
            accepted=accepted,
        )

    def _fail(self, attempt: Attempt, status: str, detail: str) -> None:
        if attempt.status != "inflight":  # a late error report changes nothing
            self._emit_late(attempt, accepted=False)
            return
        record = self._finish(attempt, status)
        if record is not None:
            record.failed += 1
            record.note_error(monotonic())
        self.reputation.record_failure(attempt.worker_id, "error")
        self.reputation.record_attempt(attempt.worker_id, True, attempt.predicted_s)
        self._emit_result(
            attempt, status, (attempt.result_t or 0) - (attempt.sent_t or 0), error=detail
        )
        self._requeue(attempt)
        self.on_progress()

    def _emit_result(
        self, attempt: Attempt, status: str, total_s: float, error: str | None = None
    ) -> None:
        self.events.emit(
            "attempt_result",
            attempt_id=attempt.attempt_id,
            worker_id=attempt.worker_id,
            status=status,
            compute_ms=attempt.compute_ms,
            queue_ms=attempt.queue_ms,
            total_s=round(total_s, 4),
            bytes_out=attempt.bytes_out,
            error=error,
        )

    def _unsolicited(self, channel: Channel, attempt_id: str) -> None:
        count = channel.unsolicited.add()
        self.security_log.emit(
            "unsolicited_result",
            worker_id=channel.worker_id,
            attempt_id=attempt_id,
            count_last_minute=count,
        )
        if count > UNSOLICITED_PER_MINUTE:
            record = self.registry.get(channel.worker_id)
            if record is not None:
                record.strikes += 1
                self.events.emit(
                    "worker_strike", worker_id=record.worker_id, strikes=record.strikes
                )
            channel.close()

    async def drain(self) -> None:
        """Wait for background sends to finish (shutdown)."""
        if self._send_tasks:
            await asyncio.gather(*list(self._send_tasks), return_exceptions=True)
