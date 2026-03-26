import os

from ghv5.serial_io import build_output_filename as _build_output_filename


def test_filename_with_dims():
    path = _build_output_filename("/tmp/data", 6.0, 4.0, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "6.0x4.0m" in name
    assert name.endswith(".csv")
    assert name.startswith("capture_")

def test_filename_without_dims():
    path = _build_output_filename("/tmp/data", None, None, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "x" not in name
    assert name.startswith("capture_")
    assert name.endswith(".csv")

def test_filename_int_dims():
    path = _build_output_filename("/tmp/data", 5.0, 3.0, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "5.0x3.0m" in name

def test_filename_includes_output_dir():
    path = _build_output_filename("/my/dir", 6.0, 4.0, timestamp="2026-01-01_120000")
    assert path.startswith("/my/dir")

def test_filename_partial_dims_falls_back():
    path = _build_output_filename("/tmp/data", 6.0, None, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "x" not in name  # partial dims must not produce "NoneX..." in name
