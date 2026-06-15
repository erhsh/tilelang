# type: ignore
"""Tests for the pass_trace debugging feature.

Covers:
- Environment variable parsing (get_pass_trace_mode)
- Diff computation (unified_diff, _count_changes)
- Programmatic pass_diff() API
- Patch / reset lifecycle
- HTML report generation
"""

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
    monkeypatch.delenv("TILELANG_PASS_TRACE", raising=False)
    monkeypatch.delenv("TILELANG_PASS_TRACE_DIR", raising=False)
    yield
    monkeypatch.delenv("TILELANG_PASS_TRACE", raising=False)
    monkeypatch.delenv("TILELANG_PASS_TRACE_DIR", raising=False)


def _simple_program():
    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            B[tid] = A[tid] + 1.0

    return program


def _noop_pass():
    return tvm.tirx.transform.Simplify()


def _transforming_pass():
    return tvm.tirx.transform.Simplify()


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------


def test_env_default_off():
    assert env.get_pass_trace_mode() is None


def test_env_off_values(monkeypatch):
    for v in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("TILELANG_PASS_TRACE", v)
        assert env.get_pass_trace_mode() is None, f"Expected None for {v!r}"


def test_env_truthy_maps_to_html(monkeypatch):
    for v in ("1", "on", "true", "yes"):
        monkeypatch.setenv("TILELANG_PASS_TRACE", v)
        assert env.get_pass_trace_mode() == "html", f"Expected 'html' for {v!r}"


def test_env_explicit_modes(monkeypatch):
    monkeypatch.setenv("TILELANG_PASS_TRACE", "terminal")
    assert env.get_pass_trace_mode() == "terminal"

    monkeypatch.setenv("TILELANG_PASS_TRACE", "html")
    assert env.get_pass_trace_mode() == "html"

    monkeypatch.setenv("TILELANG_PASS_TRACE", "both")
    assert env.get_pass_trace_mode() == "both"


def test_env_case_insensitive(monkeypatch):
    monkeypatch.setenv("TILELANG_PASS_TRACE", "TERMINAL")
    assert env.get_pass_trace_mode() == "terminal"

    monkeypatch.setenv("TILELANG_PASS_TRACE", "BOTH ")
    assert env.get_pass_trace_mode() == "both"

    monkeypatch.setenv("TILELANG_PASS_TRACE", "OFF")
    assert env.get_pass_trace_mode() is None


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def test_unified_diff_no_changes():
    from tilelang.tools.pass_trace.diff import unified_diff

    text = "line1\nline2\n"
    result = unified_diff(text, text)
    assert result == ""


def test_unified_diff_with_changes():
    from tilelang.tools.pass_trace.diff import unified_diff

    before = "line1\nline2\nline3\n"
    after = "line1\nline2_changed\nline3\n"
    result = unified_diff(before, after, color=False)
    assert "-line2" in result
    assert "+line2_changed" in result


def test_print_diff():
    from tilelang.tools.pass_trace.diff import print_diff

    before = "line1\nline2\n"
    after = "line1\nline2\n"
    assert print_diff(before, after) is False

    after = "line1\nchanged\n"
    assert print_diff(before, after) is True


# ---------------------------------------------------------------------------
# pass_diff() API
# ---------------------------------------------------------------------------


def test_pass_diff_terminal(capsys):
    from tilelang.tools.pass_trace import pass_diff

    program = _simple_program()
    results = pass_diff(program, _noop_pass(), mode="terminal")
    assert len(results) == 1
    assert "name" in results[0]
    assert "changed" in results[0]
    captured = capsys.readouterr()
    assert "Pass 1" in captured.out


def test_pass_diff_html():
    from tilelang.tools.pass_trace import pass_diff

    program = _simple_program()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name

    try:
        results = pass_diff(program, _noop_pass(), mode="html", html_path=html_path)
        assert len(results) == 1
        assert os.path.exists(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "TileLang" in content or "pass" in content.lower()
    finally:
        os.unlink(html_path)


def test_pass_diff_chained():
    from tilelang.tools.pass_trace import pass_diff

    program = _simple_program()
    passes = [
        ("Simplify1", _noop_pass()),
        ("Simplify2", _noop_pass()),
    ]
    results = pass_diff(program, passes, mode="terminal")
    assert len(results) == 2
    assert results[0]["name"] == "Simplify1"
    assert results[1]["name"] == "Simplify2"


def test_pass_diff_invalid_mode():
    from tilelang.tools.pass_trace import pass_diff

    program = _simple_program()
    with pytest.raises(ValueError, match="mode must be one of"):
        pass_diff(program, _noop_pass(), mode="invalid")


# ---------------------------------------------------------------------------
# Patch / Reset lifecycle
# ---------------------------------------------------------------------------


def test_patch_and_reset():
    from tilelang.tools.pass_trace import patch, reset

    patch()
    reset()


def test_pass_record_dataclass():
    from tilelang.tools.pass_trace import PassRecord, STATUS_COMPLETED

    rec = PassRecord(
        phase="test",
        name="TestPass",
        index=0,
        before_text="before",
        after_text="after",
        changed=True,
        add_lines=5,
        del_lines=3,
    )
    assert rec.status == STATUS_COMPLETED
    assert rec.changed is True
    assert rec.error_msg == ""


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def test_generate_html():
    from tilelang.tools.pass_trace.html import generate_html
    from tilelang.tools.pass_trace import PassRecord

    records = [
        PassRecord(
            phase="phase1_test",
            name="TestPass",
            index=0,
            before_text="x = 1\n",
            after_text="x = 2\n",
            changed=True,
            add_lines=1,
            del_lines=1,
        ),
        PassRecord(
            phase="phase1_test",
            name="NoopPass",
            index=1,
            before_text="x = 2\n",
            after_text="x = 2\n",
            changed=False,
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name

    try:
        generate_html(records, html_path)
        assert os.path.exists(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "TestPass" in content
        assert "NoopPass" in content
        assert "CHANGED" in content
        assert "NO-OP" in content
    finally:
        os.unlink(html_path)


if __name__ == "__main__":
    tilelang.testing.main()
