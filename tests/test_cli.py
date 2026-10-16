"""Tests for the compute commands of ``python -m mdc.cli`` (reference, compare, bench)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mdc import cli
from mdc.coordinator.job_builder import load_image
from mdc.tasks.image_pipeline import reference


@pytest.fixture
def image(tmp_path: Path) -> Path:
    data = np.random.default_rng(4).integers(0, 256, size=(37, 45), dtype=np.uint8)
    path = tmp_path / "in.png"
    Image.fromarray(data).save(path)
    return path


def test_reference_command(image: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ref.png"
    assert (
        cli.main(
            ["reference", "--image", str(image), "--r", "1", "--iters", "3", "--out", str(out)]
        )
        == 0
    )
    assert "written to" in capsys.readouterr().out
    np.testing.assert_array_equal(load_image(out), reference(load_image(image), 1, 3))


def test_reference_command_enforces_limits(
    image: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(["reference", "--image", str(image), "--r", "9", "--out", str(tmp_path / "x.png")])
        == 1
    )
    assert "radius" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "reference",
                "--image",
                str(tmp_path / "missing.png"),
                "--out",
                str(tmp_path / "x.png"),
            ]
        )
        == 1
    )


def test_compare_command(image: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compare", str(image), str(image)]) == 0
    assert capsys.readouterr().out.strip() == "identical"
    data = load_image(image)
    data[0, :3] ^= 1
    Image.fromarray(data).save(tmp_path / "changed.png")
    assert cli.main(["compare", str(image), str(tmp_path / "changed.png")]) == 1
    assert capsys.readouterr().out.startswith("3 differing pixels")
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(tmp_path / "small.png")
    assert cli.main(["compare", str(image), str(tmp_path / "small.png")]) == 1
    assert "different shapes" in capsys.readouterr().out


def test_bench_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bench", "--runs", "1"]) == 0
    value = float(capsys.readouterr().out.split()[0])
    assert value > 0
