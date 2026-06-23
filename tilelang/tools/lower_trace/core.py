"""IR lower trace — zero-intrusion debug tool for visualizing compilation passes.

Monkey-patches ``tvm.ir.transform.Pass.__call__`` and ``PassPipeline.lower``
to automatically capture IR before/after every pass and generate diff reports.

This module has **no dependency on ``tilelang.env``**; configuration is read
from ``os.environ`` directly, or passed programmatically via ``patch()``.

Supports two architectures:
- New: ``PassPipeline.lower`` (each backend registers a pipeline object)
- Old: phase-based functions called from ``tilelang.engine.lower``

Usage::

    TILELANG_LOWER_TRACE=1 python my_kernel.py        # HTML report
    TILELANG_LOWER_TRACE=terminal python my_kernel.py  # terminal diff only
    TILELANG_LOWER_TRACE=both python my_kernel.py      # both terminal and HTML
"""

from __future__ import annotations

import ast
import contextlib
import difflib
import dis
import functools
import inspect
import os
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class LowerRecord:
    """Result of running a single pass."""

    phase: str
    name: str
    index: int
    before_text: str
    after_text: str
    changed: bool
    add_lines: int = 0
    del_lines: int = 0
    status: str = STATUS_COMPLETED
    error_msg: str = ""


_records: list[LowerRecord] = []
_section_cache: list[str] = []
_original_pass_call: Callable | None = None
_original_pipeline_lower: object | None = None
_current_phase: str | None = None
_pass_index: int = 0
_auto_flush: bool = False
_trace_dir: str | None = None
_lock = threading.RLock()
_run_counter: int = 0
_atexit_registered: bool = False

_UNSET: object = object()
_mode_override: str | None | object = _UNSET
_trace_dir_override: str | None | object = _UNSET

# Phase label used for passes that run outside any PassPipeline.lower window
# (e.g. pre-pipeline module passes and tvm.build postproc), so they are still
# captured by the global Pass.__call__ hook.
_UNSCOPED_PHASE = "unscoped"


def _parse_lower_trace_mode(value: str | None) -> str | None:
    """Parse a TILELANG_LOWER_TRACE-style value into a mode string."""
    if value is None:
        return None
    v = value.lower().strip()
    if v in ("", "0", "false", "no", "off"):
        return None
    if v in ("1", "true", "yes", "on"):
        return "html"
    if v in ("terminal", "html", "both"):
        return v
    return "html"


def _get_mode() -> str | None:
    if _mode_override is not _UNSET:
        return _mode_override  # type: ignore[return-value]
    return _parse_lower_trace_mode(os.environ.get("TILELANG_LOWER_TRACE"))


def _is_trace_enabled() -> bool:
    return _get_mode() is not None


def _should_print_terminal() -> bool:
    mode = _get_mode()
    return mode in ("terminal", "both")


def _should_gen_html() -> bool:
    mode = _get_mode()
    return mode in ("html", "both")


def _ensure_trace_dir() -> str:
    """Initialize and return the trace directory path (created on first call)."""
    global _trace_dir

    if _trace_dir is not None:
        return _trace_dir

    from datetime import datetime

    base_dir = (
        _trace_dir_override
        if _trace_dir_override is not _UNSET and _trace_dir_override
        else os.environ.get("TILELANG_LOWER_TRACE_DIR")
        or os.path.join(".", "tmp", "lower_trace_output")
    )
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "kernel"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _trace_dir = os.path.join(base_dir, f"{script_name}_{timestamp}_{os.getpid()}")

    os.makedirs(_trace_dir, exist_ok=True)
    return _trace_dir


def _save_raw_files(record: LowerRecord):
    """Write before/after .tir files to disk (phase subdirectory layout)."""
    trace_dir = _ensure_trace_dir()
    phase_dir = os.path.join(trace_dir, record.phase)
    os.makedirs(phase_dir, exist_ok=True)

    prefix = f"{record.index:02d}_{record.name}"
    with open(os.path.join(phase_dir, f"{prefix}_before.tir"), "w") as f:
        f.write(record.before_text)
    with open(os.path.join(phase_dir, f"{prefix}_after.tir"), "w") as f:
        f.write(record.after_text)


def _get_pass_display_name(pass_obj) -> str:
    """Extract display name from pass_info.name, e.g. 'tir.Simplify' -> 'Simplify'."""
    try:
        name = str(pass_obj.info.name)
        return name.split(".")[-1] if "." in name else name
    except Exception:
        return type(pass_obj).__name__


def _incremental_flush_html():
    """Write the current HTML report incrementally.

    Uses _section_cache to avoid re-rendering previously completed sections.
    Total cost is O(n) instead of O(n^2) for full rewrites.
    """
    if not _records or not _trace_dir:
        return

    from .html import generate_html

    html_path = os.path.join(_trace_dir, "lower_trace.html")
    generate_html(_records, html_path)


def _traced_pass_call(self, mod):
    """Intercept all Pass.__call__ invocations to record before/after IR.

    Captures every pass invocation globally (matching pass_diff's hook),
    including those outside any PassPipeline.lower window (pre-pipeline module
    passes, tvm.build postproc).  Records are appended at runtime with the
    pass's actual display name, eliminating the prior index-based pre-registration
    that could drift when conditional passes (e.g. LetInline) were skipped.
    """
    global _pass_index

    if not _is_trace_enabled():
        return _original_pass_call(self, mod)

    phase = _current_phase or _UNSCOPED_PHASE
    gen_html = _should_gen_html()
    if gen_html:
        _ensure_trace_dir()
    before_text = str(mod)

    with _lock:
        idx = _pass_index
        _pass_index += 1

    try:
        result = _original_pass_call(self, mod)
    except Exception as e:
        with _lock:
            record = LowerRecord(
                phase=phase,
                name=_get_pass_display_name(self),
                index=idx,
                before_text=before_text,
                after_text="",
                changed=False,
                add_lines=0,
                del_lines=0,
                status=STATUS_FAILED,
                error_msg=str(e),
            )
            _records.append(record)
            _save_raw_files(record)
            print(f"  [lower_trace] {phase}/{idx:02d}_{record.name}: FAILED ({e})")
        raise

    after_text = str(result)
    changed = before_text != after_text

    pass_name = _get_pass_display_name(self)

    add_count = del_count = 0
    if changed:
        sm = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                add_count += j2 - j1
            elif tag == "delete":
                del_count += i2 - i1
            elif tag == "replace":
                add_count += j2 - j1
                del_count += i2 - i1

    with _lock:
        record = LowerRecord(
            phase=phase,
            name=pass_name,
            index=idx,
            before_text=before_text,
            after_text=after_text,
            changed=changed,
            add_lines=add_count,
            del_lines=del_count,
            status=STATUS_COMPLETED,
        )
        _records.append(record)
        _save_raw_files(record)
        tag = "CHANGED" if changed else "NO-OP"
        print(f"  [lower_trace] {phase}/{idx:02d}_{pass_name}: {tag}")

        if gen_html:
            with contextlib.suppress(Exception):
                _incremental_flush_html()

    if _should_print_terminal() and changed:
        from .diff import print_diff

        label = f"{phase}/{pass_name}"
        print_diff(before_text, after_text, f"{label} (before)", f"{label} (after)")

    return result


def _extract_pass_name_from_attr_chain(node: ast.expr) -> str | None:
    """Walk an attribute chain (e.g. tilelang.transform.Simplify) and extract pass name.

    Returns the pass name (e.g. 'Simplify') if the chain contains a 'transform' segment
    followed by an uppercase CamelCase name. Returns None otherwise.
    """
    if not isinstance(node, ast.Attribute):
        return None
    names = []
    cur = node
    while isinstance(cur, ast.Attribute):
        names.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        names.append(cur.id)
    names.reverse()
    try:
        transform_idx = names.index("transform")
    except ValueError:
        return None
    if transform_idx + 1 >= len(names):
        return None
    pass_name = names[transform_idx + 1]
    if not pass_name or not pass_name[0].isupper():
        return None
    return pass_name


def _discover_passes(phase_func) -> list[str]:
    """Extract pass names from a phase function's source code via AST parsing."""
    try:
        source = inspect.getsource(phase_func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    passes = []
    seen_calls: set = set()

    class _PassVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            func = node.func
            found_in_nested = False
            while isinstance(func, ast.Call):
                if id(func) not in seen_calls:
                    seen_calls.add(id(func))
                    name = _extract_pass_name_from_attr_chain(func.func)
                    if name:
                        passes.append(name)
                        found_in_nested = True
                func = func.func

            if not found_in_nested and id(node) not in seen_calls:
                seen_calls.add(id(node))
                name = _extract_pass_name_from_attr_chain(func)
                if name:
                    passes.append(name)

            self.generic_visit(node)

    _PassVisitor().visit(tree)
    return passes


def _discover_passes_recursive(phase_func) -> list[str]:
    """Extract pass names, following local helper calls in the same module."""
    passes = []
    visited = set()
    seen_calls: set = set()

    def _visit(func):
        func_id = id(func)
        if func_id in visited:
            return
        visited.add(func_id)

        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            return

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return

        func_module = inspect.getmodule(func)
        local_ns = {}
        if func_module:
            local_ns.update(vars(func_module))
        if hasattr(func, "__globals__"):
            local_ns.update(func.__globals__)

        class _PassVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                call_func = node.func

                found_in_nested = False
                while isinstance(call_func, ast.Call):
                    if id(call_func) not in seen_calls:
                        seen_calls.add(id(call_func))
                        name = _extract_pass_name_from_attr_chain(call_func.func)
                        if name:
                            passes.append(name)
                            found_in_nested = True
                    call_func = call_func.func

                if not found_in_nested and id(node) not in seen_calls:
                    seen_calls.add(id(node))
                    if isinstance(call_func, ast.Attribute):
                        name = _extract_pass_name_from_attr_chain(call_func)
                        if name:
                            passes.append(name)

                    elif isinstance(call_func, ast.Name):
                        name = call_func.id
                        resolved = local_ns.get(name)
                        if (
                            resolved is not None
                            and callable(resolved)
                            and not isinstance(resolved, type)
                            and not inspect.isbuiltin(resolved)
                        ):
                            resolved_module = getattr(resolved, "__module__", None)
                            func_module_name = getattr(func, "__module__", None)
                            if resolved_module == func_module_name:
                                _visit(resolved)

                self.generic_visit(node)

        _PassVisitor().visit(tree)

    _visit(phase_func)
    return passes


def _discover_phases(lower_func) -> list:
    """Discover phase functions from the old architecture via bytecode scanning."""
    try:
        from tilelang.engine import phase as phase_module
    except ImportError:
        return []

    phase_funcs = []
    seen_names = set()
    try:
        for instr in dis.get_instructions(lower_func):
            if instr.opname == "LOAD_GLOBAL" and instr.argval not in seen_names:
                name = instr.argval
                seen_names.add(name)
                func = getattr(phase_module, name, None)
                if func is not None and callable(func):
                    phase_funcs.append(func)
    except (TypeError, OSError):
        pass

    if not phase_funcs:
        phase_funcs = [
            getattr(phase_module, name)
            for name in sorted(dir(phase_module))
            if not name.startswith("_") and callable(getattr(phase_module, name, None))
        ]

    def _src_line(f):
        try:
            return inspect.getsourcelines(f)[1]
        except (OSError, TypeError):
            return 999999

    phase_funcs.sort(key=_src_line)
    return phase_funcs


def _wrap_phase(original_func, phase_index, total_phases):
    """Wrap a phase function to set tracing context (legacy architecture).

    Phase context is set so that passes invoked within this window are tagged
    with the phase label.  Pass records are appended at runtime by
    ``_traced_pass_call``; no pre-registration is performed.
    """
    base_phase_name = f"phase{phase_index}_{original_func.__name__}"

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        global _run_counter, _current_phase, _auto_flush

        with _lock:
            if phase_index == 1:
                _run_counter += 1
                if _run_counter == 1:
                    reset()

            run_prefix = f"run{_run_counter}_" if _run_counter > 1 else ""
            phase_name = f"{run_prefix}{base_phase_name}"

            _current_phase = phase_name
            _auto_flush = _should_gen_html()

        try:
            result = original_func(*args, **kwargs)
        except Exception as e:
            with _lock:
                _auto_flush = False
                _current_phase = None
                print(f"  [lower_trace] EXCEPTION in {phase_name}: {e}")

            with contextlib.suppress(Exception):
                _incremental_flush_html()

            raise

        with _lock:
            _auto_flush = False
            _current_phase = None

            if phase_index == total_phases:
                print(f"  [lower_trace] run {_run_counter} ({phase_name}) complete: {len(_records)} total records")

        with contextlib.suppress(Exception):
            _incremental_flush_html()

        return result

    return wrapper


def _traced_pipeline_lower(self, mod, target):
    """Intercept PassPipeline.lower to set phase context for pass tracing (new architecture).

    Only sets ``_current_phase`` so that passes invoked within this window are
    tagged with the pipeline label.  Pass records are appended at runtime by
    ``_traced_pass_call``; no pre-registration is performed, so conditional
    passes (e.g. LetInline) that are skipped at runtime simply do not appear,
    matching the behaviour of ``pass_diff``.
    """
    global _run_counter, _current_phase, _auto_flush

    with _lock:
        _run_counter += 1
        if _run_counter == 1:
            reset()

        run_prefix = f"run{_run_counter}_" if _run_counter > 1 else ""
        phase_name = f"{run_prefix}pipeline_{self.name}"
        _current_phase = phase_name
        _auto_flush = _should_gen_html()

    try:
        result = _original_pipeline_lower(self, mod, target)
    except Exception as e:
        with _lock:
            _auto_flush = False
            _current_phase = None
            print(f"  [lower_trace] EXCEPTION in {phase_name}: {e}")

        with contextlib.suppress(Exception):
            _incremental_flush_html()

        raise

    with _lock:
        _auto_flush = False
        _current_phase = None

    with contextlib.suppress(Exception):
        _incremental_flush_html()

    print(f"  [lower_trace] run {_run_counter} ({phase_name}) complete: {len(_records)} total records")

    return result


def _register_atexit():
    """Register the final-report atexit handler (idempotent)."""
    global _atexit_registered
    if _atexit_registered:
        return
    import atexit

    atexit.register(_final_report)
    _atexit_registered = True


def patch(*, mode=_UNSET, trace_dir=_UNSET):
    """Activate IR pass tracing via monkey-patching.

    Parameters
    ----------
    mode : str | None, optional
        Force a trace mode (``'terminal'``, ``'html'``, ``'both'``, or
        ``None`` to disable).  When omitted, the mode is read from the
        ``TILELANG_LOWER_TRACE`` env var (or a prior ``patch`` override),
        keeping this module free of any ``tilelang.env`` dependency.
    trace_dir : str | None, optional
        Force the trace output base directory.  When omitted, falls back to
        the ``TILELANG_LOWER_TRACE_DIR`` env var, then
        ``./tmp/lower_trace_output``.
    """
    global _mode_override, _trace_dir_override

    if mode is not _UNSET:
        _mode_override = _parse_lower_trace_mode(mode if mode is None else str(mode))
    if trace_dir is not _UNSET:
        _trace_dir_override = trace_dir if trace_dir is None else str(trace_dir)

    from tvm.ir.transform import Pass

    global _original_pass_call, _original_pipeline_lower, _atexit_registered
    if _original_pass_call is None:
        _original_pass_call = Pass.__call__
        Pass.__call__ = _traced_pass_call

    _register_atexit()

    if _original_pipeline_lower is not None:
        return

    try:
        from tilelang.backend.pass_pipeline import PassPipeline

        _original_pipeline_lower = PassPipeline.lower
        PassPipeline.lower = _traced_pipeline_lower
        print("[lower_trace] IR pass tracing patched (PassPipeline architecture). Set TILELANG_LOWER_TRACE=1 to enable.")
        return
    except ImportError:
        pass

    try:
        import tilelang.engine.lower as lower_mod

        lower_func = lower_mod.lower
        patch_mod = lower_mod
    except (ImportError, AttributeError):
        try:
            from tilelang.engine import lower as lower_func

            import tilelang.engine as patch_mod
        except (ImportError, AttributeError) as e:
            print(f"[lower_trace] WARNING: could not patch — {e}")
            return

    phase_funcs = _discover_phases(lower_func)
    for i, phase_func in enumerate(phase_funcs):
        wrapped = _wrap_phase(phase_func, i + 1, len(phase_funcs))
        setattr(patch_mod, phase_func.__name__, wrapped)
        try:
            from tilelang.engine import phase as phase_module

            if hasattr(phase_module, phase_func.__name__):
                setattr(phase_module, phase_func.__name__, wrapped)
        except ImportError:
            pass
        if phase_func.__name__ in getattr(lower_func, "__globals__", {}):
            lower_func.__globals__[phase_func.__name__] = wrapped

    _original_pipeline_lower = True
    print(
        f"[lower_trace] IR pass tracing patched (phase-based architecture, "
        f"{len(phase_funcs)} phases). Set TILELANG_LOWER_TRACE=1 to enable."
    )


def _final_report():
    """Generate final HTML report at process exit, covering all accumulated runs."""
    if not _records or not _trace_dir:
        return
    try:
        from .html import generate_html

        html_path = os.path.join(_trace_dir, "lower_trace.html")
        generate_html(_records, html_path)
        print(f"  [lower_trace] Final HTML report: {html_path}")
    except Exception as exc:
        print(f"  [lower_trace] WARNING: failed to generate final HTML report: {exc}")


def uninstall():
    """Remove the pass tracing hook and restore original behavior."""
    global _original_pass_call, _original_pipeline_lower, _atexit_registered, _run_counter
    global _mode_override, _trace_dir_override

    if _original_pass_call is not None:
        from tvm.ir.transform import Pass

        Pass.__call__ = _original_pass_call
        _original_pass_call = None

    if _original_pipeline_lower is not None and _original_pipeline_lower is not True:
        from tilelang.backend.pass_pipeline import PassPipeline

        PassPipeline.lower = _original_pipeline_lower

    _original_pipeline_lower = None

    _mode_override = _UNSET
    _trace_dir_override = _UNSET

    if _atexit_registered:
        import atexit

        atexit.unregister(_final_report)
        _atexit_registered = False

    _run_counter = 0
    reset()


def reset():
    """Clear collected records, section cache, and trace directory."""
    global _records, _section_cache, _trace_dir, _current_phase, _pass_index, _auto_flush
    _records = []
    _section_cache = []
    _trace_dir = None
    _current_phase = None
    _pass_index = 0
    _auto_flush = False
