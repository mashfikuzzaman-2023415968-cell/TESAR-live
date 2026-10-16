"""Tests for the job DAG (spec 7.2) and the job builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mdc.common.config import load_config
from mdc.coordinator import job_builder as JB
from mdc.coordinator.job import (
    FINAL,
    Attempt,
    Job,
    TaskNode,
    TaskStatus,
    execute_inprocess,
    result_hash,
)
from mdc.tasks.base import execute
from mdc.tasks.image_pipeline import reference
from mdc.tasks.synthetic import hash_chain, xor_digests


def small_job(h: int = 20, w: int = 30, tile: int = 10, r: int = 1, iters: int = 1) -> Job:
    img = np.random.default_rng(0).integers(0, 256, size=(h, w), dtype=np.uint8)
    return JB.build_image_job_from_array(img, tile, r, iters, 64, job_id="job_t")


def run_tasks(job: Job, task_ids: list[str], worker: str = "w1") -> Attempt:
    """Execute one attempt for ``task_ids`` and apply its result."""
    attempt = Attempt(attempt_id=f"a_{len(job.attempts)}", task_ids=task_ids, worker_id=worker)
    job.start_attempt(attempt)
    kind, params, data = job.task_input(task_ids)
    out = execute(kind, params, data)
    attempt.status = "ok"
    if out.array is not None:
        job.apply_chunk_output(task_ids, out.array, worker)
    else:
        job.mark_done(task_ids[0], out.json_result, worker)
    return attempt


# ----------------------------------------------------------- structure


def test_image_job_structure() -> None:
    job = small_job()
    assert len(job.stage1_ids()) == 6  # 2 rows x 3 cols
    assert job.tasks["s2_r01"].deps == ["s1_r01_c00", "s1_r01_c01", "s1_r01_c02"]
    assert job.tasks[FINAL].deps == ["s2_r00", "s2_r01"]
    assert job.tasks["s1_r00_c00"].params == {"r": 1, "iters": 1}
    assert job.tasks["s2_r00"].params == {"edge_threshold": 64}
    assert job.params["halo"] == 2
    assert job.tasks["s1_r00_c00"].input_bytes_est == 14 * 14
    assert job.tasks["s1_r00_c00"].output_bytes_est == 100
    assert job.tasks["s1_r00_c00"].work_units > 0
    assert job.dependents("s1_r00_c00") == ["s2_r00"]


def test_initial_readiness_and_order() -> None:
    job = small_job()
    ready = [t.task_id for t in job.ready_tasks()]
    assert ready == job.stage1_ids()  # row-major
    assert job.tasks["s2_r00"].status is TaskStatus.PENDING
    assert job.tasks[FINAL].status is TaskStatus.PENDING
    assert not job.final_ready()
    assert not job.is_complete()


def test_row_becomes_ready_and_stage2_scheduled_first() -> None:
    job = small_job()
    run_tasks(job, ["s1_r00_c00", "s1_r00_c01"])
    assert job.tasks["s2_r00"].status is TaskStatus.PENDING  # one tile of the row missing
    run_tasks(job, ["s1_r00_c02"])
    assert job.tasks["s2_r00"].status is TaskStatus.READY
    assert job.ready_tasks()[0].task_id == "s2_r00"  # stage 2 before remaining stage 1


def test_full_job_completes_and_matches_reference() -> None:
    job = small_job(37, 53, 8, 2, 2)
    summary = execute_inprocess(job, [2, 1, 3])
    assert job.is_complete()
    img = job.padded[
        job.params["halo"] : -job.params["halo"], job.params["halo"] : -job.params["halo"]
    ]  # type: ignore[index]
    assert summary["output"].tobytes() == reference(img, 2, 2).tobytes()
    assert summary["pixels"] == 37 * 53
    assert summary["edge_density"] == summary["edge_pixels"] / summary["pixels"]
    assert job.tasks[FINAL].result["pixels"] == 37 * 53
    with pytest.raises(ValueError, match="not ready"):
        job.run_final()


def test_mark_done_is_idempotent() -> None:
    job = small_job()
    first = np.zeros((10, 10), dtype=np.uint8)
    assert job.mark_done("s1_r00_c00", first, "w1") is True
    assert job.mark_done("s1_r00_c00", np.ones((10, 10), dtype=np.uint8), "w2") is False
    task = job.tasks["s1_r00_c00"]
    assert task.produced_by == "w1"
    assert task.result_sha256 == result_hash(first)


def test_failed_task_not_marked_done() -> None:
    job = small_job()
    job.tasks["s1_r00_c00"].status = TaskStatus.FAILED
    assert job.mark_done("s1_r00_c00", np.zeros((10, 10), dtype=np.uint8), "w") is False


# -------------------------------------------------------- invalidation


def complete_job() -> Job:
    job = small_job()
    execute_inprocess(job)
    return job


def test_invalidate_tile_cascades_to_row_stats_and_final() -> None:
    job = complete_job()
    affected = job.invalidate("s1_r00_c01")
    assert affected[0] == "s1_r00_c01"
    assert set(affected) == {"s1_r00_c01", "s2_r00", FINAL}
    assert job.tasks["s1_r00_c01"].status is TaskStatus.INVALIDATED
    assert job.tasks["s1_r00_c01"].result is None
    assert job.tasks["s2_r00"].status is TaskStatus.PENDING
    assert job.tasks[FINAL].status is TaskStatus.PENDING
    # The other row is untouched (partial dependencies).
    assert job.tasks["s2_r01"].status is TaskStatus.DONE
    assert not job.is_complete()
    assert [t.task_id for t in job.ready_tasks()] == ["s1_r00_c01"]
    # Re-running repairs everything.
    run_tasks(job, ["s1_r00_c01"])
    run_tasks(job, ["s2_r00"])
    job.run_final()
    assert job.is_complete()


def test_invalidate_with_pending_dependents_stops_early() -> None:
    job = small_job()
    run_tasks(job, ["s1_r00_c00"])
    assert job.invalidate("s1_r00_c00") == ["s1_r00_c00"]  # s2_r00 never ran


def test_stale_attempt_detected_by_epoch() -> None:
    job = small_job()
    for tid in ["s1_r00_c00", "s1_r00_c01", "s1_r00_c02"]:
        run_tasks(job, [tid])
    stage2 = Attempt(attempt_id="a_s2", task_ids=["s2_r00"], worker_id="w2")
    job.start_attempt(stage2)
    assert job.tasks["s2_r00"].status is TaskStatus.RUNNING
    assert job.attempt_is_current(stage2)
    job.invalidate("s1_r00_c02")  # its input changed while it was running
    assert not job.attempt_is_current(stage2)
    assert job.tasks["s2_r00"].status is TaskStatus.PENDING


def test_replace_result_invalidates_dependents_only() -> None:
    job = complete_job()
    corrected = np.full((10, 10), 7, dtype=np.uint8)
    job.replace_result("s1_r01_c00", corrected, "local")
    tile = job.tasks["s1_r01_c00"]
    assert tile.status is TaskStatus.DONE and tile.verified and tile.produced_by == "local"
    assert job.tasks["s2_r01"].status is TaskStatus.READY  # recompute with the new tile
    assert job.tasks[FINAL].status is TaskStatus.PENDING
    run_tasks(job, ["s2_r01"])
    summary = job.run_final()
    assert (summary["output"][10:20, 0:10] == 7).all()


# ------------------------------------------------------------ attempts


def test_start_attempt_rules() -> None:
    job = small_job()
    with pytest.raises(ValueError, match="at least one"):
        job.start_attempt(Attempt("a0", [], "w"))
    with pytest.raises(ValueError, match="never sent"):
        job.start_attempt(Attempt("a1", [FINAL], "w"))
    with pytest.raises(ValueError, match="not runnable"):
        job.start_attempt(Attempt("a2", ["s2_r00"], "w"))  # still PENDING
    job.start_attempt(Attempt("a3", ["s1_r00_c00"], "w"))
    with pytest.raises(ValueError, match="duplicate"):
        job.start_attempt(Attempt("a3", ["s1_r00_c01"], "w"))
    assert job.attempts["a3"].sent_t is not None
    assert job.tasks["s1_r00_c00"].attempts == ["a3"]


def test_replica_and_requeue() -> None:
    job = small_job()
    primary = Attempt("p", ["s1_r00_c00"], "w1")
    replica = Attempt("r", ["s1_r00_c00"], "w2", purpose="replica")
    job.start_attempt(primary)
    job.start_attempt(replica)
    assert len(job.inflight_attempts("s1_r00_c00")) == 2
    primary.status = "lost"
    assert job.requeue("s1_r00_c00") is False  # the replica is still in flight
    replica.status = "timeout"
    assert job.requeue("s1_r00_c00") is True
    assert job.tasks["s1_r00_c00"].status is TaskStatus.READY
    assert job.requeue("s1_r00_c00") is False  # not RUNNING any more


def test_verify_attempt_does_not_change_status() -> None:
    job = complete_job()
    job.start_attempt(Attempt("v", ["s1_r00_c00"], "local", purpose="verify"))
    assert job.tasks["s1_r00_c00"].status is TaskStatus.DONE


@pytest.mark.parametrize(
    "task_ids",
    [
        ["s1_r00_c00", "s1_r00_c02"],  # gap
        ["s1_r00_c00", "s1_r01_c00"],  # two rows
        ["s1_r00_c00", "s2_r00"],  # mixed kinds
        [],
    ],
)
def test_chunk_must_be_contiguous_run_in_one_row(task_ids: list[str]) -> None:
    with pytest.raises(ValueError):
        small_job().chunk_columns(task_ids)


def test_task_input_and_expected_shapes() -> None:
    job = small_job()
    kind, params, data = job.task_input(["s1_r01_c01", "s1_r01_c02"])
    assert kind == "img_tile" and params == {"r": 1, "iters": 1}
    assert data is not None and data.shape == (10 + 4, 20 + 4)
    assert job.expected_output_shape(["s1_r01_c01", "s1_r01_c02"]) == (10, 20)
    assert job.expected_output_shape(["s2_r00"]) is None
    with pytest.raises(KeyError):
        job.task_input(["s2_r00"])  # its row has no tile results yet
    with pytest.raises(ValueError, match="never chunked"):
        job.task_input(["s2_r00", "s2_r01"])


def test_apply_chunk_output_checks_shape() -> None:
    job = small_job()
    with pytest.raises(Exception, match="shape"):
        job.apply_chunk_output(["s1_r00_c00"], np.zeros((5, 5), dtype=np.uint8), "w")


def test_invalid_graphs_rejected() -> None:
    a = TaskNode("a", 1, "synthetic", ["b"])
    b = TaskNode("b", 1, "synthetic", ["a"])
    with pytest.raises(ValueError, match="cycle"):
        Job("j", "synthetic", {"a": a, "b": b}, {})
    with pytest.raises(ValueError, match="unknown task"):
        Job("j", "synthetic", {"a": TaskNode("a", 1, "synthetic", ["zzz"])}, {})


def test_status_counts() -> None:
    counts = small_job().status_counts()
    assert counts["ready"] == 6 and counts["pending"] == 3
    assert sum(counts.values()) == 9


def test_image_job_without_geometry_errors() -> None:
    job = Job(
        "j", "image", {"s1_r00_c00": TaskNode("s1_r00_c00", 1, "img_tile", [], row=0, col=0)}, {}
    )
    with pytest.raises(ValueError, match="geometry"):
        job.task_input(["s1_r00_c00"])
    with pytest.raises(ValueError):
        job.row_strip(0)
    with pytest.raises(ValueError):
        job.apply_chunk_output(["s1_r00_c00"], np.zeros((1, 1), dtype=np.uint8), "w")


# ------------------------------------------------------- synthetic jobs


def test_synthetic_job() -> None:
    job = JB.build_synthetic_job(5, 20, seed=42)
    assert [t.task_id for t in job.ready_tasks()] == [f"syn_{i:04d}" for i in range(5)]
    assert job.tasks["syn_0003"].params == {"seed": 42 * 1_000_003 + 3, "chain_len": 20}
    assert job.tasks["syn_0000"].work_units == pytest.approx(0.02)
    summary = execute_inprocess(job, [4])  # chunk sizes do not apply to synthetic tasks
    expected = xor_digests([hash_chain(42 * 1_000_003 + i, 20) for i in range(5)])
    assert summary == {"digest": expected}
    assert job.is_complete()
    with pytest.raises(ValueError):
        JB.build_synthetic_job(0, 10, 1)


def test_synthetic_job_from_config() -> None:
    cfg = load_config(
        overrides=["job.type=synthetic", "job.synthetic_tasks=3", "job.synthetic_chain_len=10"]
    )
    job = JB.build_job(cfg)
    assert job.job_type == "synthetic" and len(job.tasks) == 4


# ---------------------------------------------------------- job builder


def test_build_image_job_from_config(repo_root: Path) -> None:
    cfg = load_config(
        overrides=[
            f"job.image_path={repo_root}/data/images/test_512.png",
            "job.base_tile=128",
            "job.blur_iterations=2",
        ]
    )
    job = JB.build_job(cfg, job_id="cfgjob")
    assert job.job_id == "cfgjob"
    assert job.grid is not None and (job.grid.rows, job.grid.cols) == (4, 4)
    assert job.image_path is not None and job.image_path.endswith("test_512.png")


def test_build_image_job_rejects_non_uint8() -> None:
    with pytest.raises(ValueError):
        JB.build_image_job_from_array(np.zeros((8, 8), dtype=np.int32), 4, 1, 1, 64)


def test_load_and_save_image(tmp_path: Path) -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    gray = JB.load_image(tmp_path / "rgb.png")
    assert gray.shape == (6, 8) and gray.dtype == np.uint8
    assert gray.flags["WRITEABLE"]
    JB.save_image(tmp_path / "sub" / "out.png", gray)
    np.testing.assert_array_equal(JB.load_image(tmp_path / "sub" / "out.png"), gray)


def test_load_image_rejects_bad_files(tmp_path: Path) -> None:
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(JB.ImageLoadError, match="cannot read"):
        JB.load_image(tmp_path / "x.png")
    with pytest.raises(JB.ImageLoadError):
        JB.load_image(tmp_path / "missing.png")


def test_load_image_rejects_oversized(tmp_path: Path) -> None:
    Image.new("L", (8193, 1)).save(tmp_path / "wide.png")
    with pytest.raises(JB.ImageLoadError, match="exceeds"):
        JB.load_image(tmp_path / "wide.png")


def test_load_image_rejects_decompression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new("L", (300, 300)).save(tmp_path / "big.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # 90 000 px > 2 x limit -> bomb
    with pytest.raises(JB.ImageLoadError, match="bomb"):
        JB.load_image(tmp_path / "big.png")


def test_result_hash_kinds() -> None:
    assert result_hash({"b": 1, "a": 2}) == result_hash({"a": 2, "b": 1})  # canonical
    assert result_hash(np.zeros(3, dtype=np.uint8)) != result_hash(np.ones(3, dtype=np.uint8))
