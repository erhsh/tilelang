# type: ignore
"""Tests for the lower_trace debugging feature."""

import os
import pytest
import tempfile

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.env import env


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("TILELANG_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TILELANG_LOWER_TRACE_DIR", raising=False)
    yield
    monkeypatch.delenv("TILELANG_LOWER_TRACE", raising=False)
    monkeypatch.delenv("TILELANG_LOWER_TRACE_DIR", raising=False)


def _simple_program():
    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return program


def _noop_pass():
    return tvm.tirx.transform.Simplify()


def test_env_default_off():
    assert env.get_lower_trace_mode() is None


def test_env_off_values(monkeypatch):
    for v in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("TILELANG_LOWER_TRACE", v)
        assert env.get_lower_trace_mode() is None, f"Expected None for {v!r}"


def test_env_truthy_maps_to_html(monkeypatch):
    for v in ("1", "on", "true", "yes"):
        monkeypatch.setenv("TILELANG_LOWER_TRACE", v)
        assert env.get_lower_trace_mode() == "html", f"Expected 'html' for {v!r}"


def test_env_explicit_modes(monkeypatch):
    monkeypatch.setenv("TILELANG_LOWER_TRACE", "terminal")
    assert env.get_lower_trace_mode() == "terminal"

    monkeypatch.setenv("TILELANG_LOWER_TRACE", "html")
    assert env.get_lower_trace_mode() == "html"

    monkeypatch.setenv("TILELANG_LOWER_TRACE", "both")
    assert env.get_lower_trace_mode() == "both"


def test_lower_trace_api_single_pass(capsys):
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    results = lower_trace(program, _noop_pass(), mode="terminal")
    assert len(results) == 1
    assert "name" in results[0]
    assert "changed" in results[0]
    captured = capsys.readouterr()
    assert "Pass 1" in captured.out


def test_lower_trace_api_chain():
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    passes = [
        ("Simplify1", _noop_pass()),
        ("Simplify2", _noop_pass()),
    ]
    results = lower_trace(program, passes, mode="terminal")
    assert len(results) == 2
    assert results[0]["name"] == "Simplify1"
    assert results[1]["name"] == "Simplify2"


def test_patch_uninstall():
    from tilelang.tools.lower_trace import patch, uninstall

    patch()
    uninstall()


def test_lower_trace_html():
    from tilelang.tools.lower_trace import lower_trace

    program = _simple_program()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name

    try:
        results = lower_trace(program, _noop_pass(), mode="html", html_path=html_path)
        assert len(results) == 1
        assert os.path.exists(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "TileLang" in content or "pass" in content.lower()
    finally:
        os.unlink(html_path)


def test_discover_passes():
    from tilelang.tools.lower_trace.core import _discover_passes
    from tilelang.cpu.pipeline import CPUPassPipelineBody

    pass_names = _discover_passes(CPUPassPipelineBody)
    assert len(pass_names) > 10, f"Expected >10 passes, got {len(pass_names)}"
    assert "Simplify" in pass_names
    assert "LayoutInference" in pass_names
    assert "BindTarget" in pass_names


if __name__ == "__main__":
    tilelang.testing.main()
