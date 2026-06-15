from __future__ import annotations
import ast
import os
import sys
import dis
import difflib
import functools
import inspect
import contextlib
import threading
from dataclasses import dataclass


STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class PassRecord:
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


# ---------------------------------------------------------------------------
# Global state: records collected during compilation
# ---------------------------------------------------------------------------
_records: list[PassRecord] = []

# Pass.__call__ interception state
_original_pass_call = None  # Saved original Pass.__call__ (None = not yet patched)
_original_pipeline_lower = None  # Saved original PassPipeline.lower (None = not yet patched)
_current_phase: str | None = None  # Active phase name during execution
_pass_index: int = 0  # Auto-incrementing pass counter within a phase
_phase_call_count: int = 0  # Tracks which phase is executing (1=first, 2=second, ...)
_num_phases: int = 0  # Total number of phases discovered at patch time
_current_pass_index: int = -1  # Index of the currently executing pass in _records
_failed_pass_info: tuple | None = None  # (before_text, error_msg) when a pass fails
_records_offset: int = 0  # Start index in _records for the current phase
_auto_flush: bool = False  # When True, write HTML after each pass (survives segfaults)

# ---------------------------------------------------------------------------
# Dump directory
# ---------------------------------------------------------------------------
_trace_dir: str | None = None

# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
_trace_lock = threading.RLock()


def _flush_html():
    """Write the current HTML report incrementally.

    Called after each successful pass when _auto_flush is True.  This ensures
    the HTML report survives process-level crashes (e.g. SIGSEGV) that bypass
    Python's exception handling.
    """
    if not _records or not _trace_dir:
        return
    from .html import generate_html

    html_path = os.path.join(_trace_dir, "ir_trace.html")
    generate_html(_records, html_path)


# ---------------------------------------------------------------------------
# Dump control
# ---------------------------------------------------------------------------
def _get_mode() -> str | None:
    from tilelang.env import env

    return env.get_pass_trace_mode()


def _is_trace_enabled() -> bool:
    return _get_mode() is not None


def _should_print_terminal() -> bool:
    mode = _get_mode()
    return mode in ("terminal", "both")


def _should_gen_html() -> bool:
    mode = _get_mode()
    return mode in ("html", "both")



# ---------------------------------------------------------------------------
# Dump directory initialization (lazy, once per compilation)
# ---------------------------------------------------------------------------
def _ensure_trace_dir() -> str:
    """Initialize and return the trace directory path (created on first call).

    Default path: ./tmp/pass_trace_dir/{kernel_name}_YYYYMMDDHHmmSS/
    Override with TILELANG_PASS_TRACE_DIR env var.
    """
    global _trace_dir

    if _trace_dir is not None:
        return _trace_dir

    from tilelang.env import env
    from datetime import datetime

    base_dir = env.TILELANG_PASS_TRACE_DIR or os.path.join(".", "tmp", "pass_trace_dir")
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "kernel"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S_%f")
    _trace_dir = os.path.join(base_dir, f"{script_name}_{timestamp}_{os.getpid()}")

    os.makedirs(_trace_dir, exist_ok=True)
    return _trace_dir


# ---------------------------------------------------------------------------
# Core: run_pass
# ---------------------------------------------------------------------------
def run_pass(pass_obj, mod, pass_name: str, phase_name: str, pass_index: int):
    """Execute a single pass, capturing before/after IR.

    Args:
        pass_obj:   A TVM Pass object (result of pass_factory())
        mod:        Current IRModule
        pass_name:  Human-readable pass name for display
        phase_name: Phase identifier (e.g. "phase1_LowerAndLegalize")
        pass_index: Sequential index within the phase

    Returns:
        The transformed IRModule (result of pass_obj(mod))
    """
    global _records

    should_trace = _is_trace_enabled()

    if should_trace:
        _ensure_trace_dir()
        before_text = str(mod)
    else:
        before_text = ""

    # Execute the actual pass
    mod = pass_obj(mod)

    if should_trace:
        after_text = str(mod)
        changed = before_text != after_text

        add_count = 0
        del_count = 0
        if changed:
            # Compute add/del counts via SequenceMatcher
            sm = difflib.SequenceMatcher(None, before_text.splitlines(), after_text.splitlines())
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "insert":
                    add_count += j2 - j1
                elif tag == "delete":
                    del_count += i2 - i1
                elif tag == "replace":
                    add_count += j2 - j1
                    del_count += i2 - i1

        record = PassRecord(
            phase=phase_name,
            name=pass_name,
            index=pass_index,
            before_text=before_text,
            after_text=after_text,
            changed=changed,
            add_lines=add_count,
            del_lines=del_count,
        )
        _records.append(record)

        # Also write raw .tir files (useful for BeyondCompare)
        _save_raw_files(record)

        # Console progress
        tag = "CHANGED" if changed else "NO-OP"
        print(f"  [pass_trace] {phase_name}/{pass_index:02d}_{pass_name}: {tag}")

    return mod


def _save_raw_files(record: PassRecord):
    """Write before/after .tir files to disk (phase subdirectory layout)."""
    trace_dir = _trace_dir
    if not trace_dir:
        return

    phase_dir = os.path.join(trace_dir, record.phase)
    os.makedirs(phase_dir, exist_ok=True)

    prefix = f"{record.index:02d}_{record.name}"
    with open(os.path.join(phase_dir, f"{prefix}_before.tir"), "w") as f:
        f.write(record.before_text)
    with open(os.path.join(phase_dir, f"{prefix}_after.tir"), "w") as f:
        f.write(record.after_text)


# ---------------------------------------------------------------------------
# Pass.__call__ interception (automatic — no pass list needed)
# ---------------------------------------------------------------------------
def _get_pass_display_name(pass_obj) -> str:
    """Extract display name from pass_info.name, e.g. 'tir.Simplify' → 'Simplify'."""
    try:
        name = str(pass_obj.info.name)
        return name.split(".")[-1] if "." in name else name
    except Exception:
        return type(pass_obj).__name__


def _traced_pass_call(self, mod):
    """Intercept all Pass.__call__ invocations to record before/after IR.

    Replaces tvm.ir.transform.Pass.__call__ at runtime.  When no phase
    context is active (normal compilation without tracing), it simply
    delegates to the original __call__ with zero overhead.

    When a phase has pre-registered pass records (via _wrap_phase), this
    function updates the existing record in-place rather than appending.
    If the pass throws, the record is marked as FAILED and the error is
    captured for the HTML report.
    """
    global _pass_index, _current_pass_index, _failed_pass_info

    if not _current_phase or not _is_trace_enabled():
        return _original_pass_call(self, mod)

    gen_html = _should_gen_html()
    if gen_html:
        _ensure_trace_dir()
    before_text = str(mod)

    with _trace_lock:
        _current_pass_index = _pass_index
        _pass_index += 1
        _failed_pass_info = None
        _rec_idx = _records_offset + _current_pass_index

    try:
        result = _original_pass_call(self, mod)
    except Exception as e:
        with _trace_lock:
            _failed_pass_info = (before_text, str(e))
            if gen_html and 0 <= _rec_idx < len(_records):
                rec = _records[_rec_idx]
                rec.status = STATUS_FAILED
                rec.before_text = before_text
                rec.error_msg = str(e)
            _current_pass_index = -1
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

    with _trace_lock:
        if gen_html:
            if 0 <= _rec_idx < len(_records):
                rec = _records[_rec_idx]
                rec.before_text = before_text
                rec.after_text = after_text
                rec.changed = changed
                rec.add_lines = add_count
                rec.del_lines = del_count
                rec.status = STATUS_COMPLETED
                _save_raw_files(rec)
                tag = "CHANGED" if changed else "NO-OP"
                print(f"  [pass_trace] {_current_phase}/{rec.index:02d}_{rec.name}: {tag}")
            else:
                record = PassRecord(
                    phase=_current_phase,
                    name=pass_name,
                    index=_current_pass_index,
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
                print(f"  [pass_trace] {_current_phase}/{record.index:02d}_{pass_name}: {tag}")

        _current_pass_index = -1

        if _auto_flush:
            with contextlib.suppress(Exception):
                _flush_html()

    if _should_print_terminal() and changed:
        from .diff import print_diff

        label = f"{_current_phase}/{pass_name}"
        print_diff(before_text, after_text, f"{label} (before)", f"{label} (after)")

    return result


# ---------------------------------------------------------------------------
# Phase wrappers (generic — auto-numbered, no hardcoded names)
# ---------------------------------------------------------------------------
def _wrap_phase(original_func, phase_index, total_phases):
    """Wrap a phase function to set tracing context.

    - phase_index: 1-based position among all phases (1=first, 2=second, ...)
    - total_phases: total number of phases in the compilation pipeline

    Before execution, discovers all pass names via AST parsing and pre-registers
    them as 'skipped'.  As each pass completes, _traced_pass_call updates the
    record to 'completed'.  If a pass throws, the record is marked 'failed' and
    remaining passes stay 'skipped'.  HTML report is generated regardless.
    """
    phase_name = f"phase{phase_index}_{original_func.__name__}"

    # Discover passes via AST (done once at wrap time, not per-call)
    pass_names = _discover_passes(original_func)

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        global _current_phase, _pass_index, _phase_call_count, _current_pass_index, _failed_pass_info, _records_offset, _auto_flush

        with _trace_lock:
            _phase_call_count += 1

            if phase_index == 1:
                reset()

            _current_phase = phase_name
            _pass_index = 0
            _current_pass_index = -1
            _failed_pass_info = None

            gen_html = _should_gen_html()

            _records_offset = len(_records)

            if gen_html and pass_names:
                _ensure_trace_dir()
                for i, name in enumerate(pass_names):
                    _records.append(
                        PassRecord(
                            phase=phase_name,
                            name=name,
                            index=i,
                            before_text="",
                            after_text="",
                            changed=False,
                            status=STATUS_SKIPPED,
                        )
                    )

            _auto_flush = gen_html

        try:
            result = original_func(*args, **kwargs)
        except Exception as e:
            with _trace_lock:
                _auto_flush = False
                _current_phase = None
                print(f"  [pass_trace] EXCEPTION in {phase_name}: {e}")

                if _records and _trace_dir:
                    try:
                        from .html import generate_html

                        html_path = os.path.join(_trace_dir, "ir_trace.html")
                        generate_html(_records, html_path)
                        print(f"  [pass_trace] HTML report (with failures) written to: {html_path}")
                    except Exception as html_err:
                        print(f"  [pass_trace] WARNING: failed to generate HTML report: {html_err}")
                        import traceback

                        traceback.print_exc()

            raise

        with _trace_lock:
            _auto_flush = False
            _current_phase = None

            if phase_index == total_phases and _records and _trace_dir:
                from .html import generate_html

                html_path = os.path.join(_trace_dir, "ir_trace.html")
                generate_html(_records, html_path)
                print(f"  [pass_trace] HTML report written to: {html_path}")

            if phase_index == total_phases:
                _phase_call_count = 0

        return result

    return wrapper


# ---------------------------------------------------------------------------
# Monkey-patch entry point
# ---------------------------------------------------------------------------
def _discover_passes_recursive(phase_func) -> list[str]:
    """Extract pass names from a pipeline function, following local helper calls.

    Like ``_discover_passes``, but when the function calls another local
    function defined in the same module (e.g. ``CUDAPassPipelineBody``
    calling ``CUDAPassPipelineBodyPrologue``), it follows into that
    function and collects its passes too.

    This is needed because the current codebase uses a ``PassPipeline``
    architecture where a single pipeline body function (like
    ``CUDAPassPipelineBody``) delegates to helper functions (like
    ``CUDAPassPipelineBodyPrologue``) that contain many of the passes.
    """
    passes = []
    visited = set()

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

        # Build local namespace for resolving function references
        func_module = inspect.getmodule(func)
        local_ns = {}
        if func_module:
            local_ns.update(vars(func_module))
        if hasattr(func, "__globals__"):
            local_ns.update(func.__globals__)

        class _PassVisitor(ast.NodeVisitor):
            """Walk the AST collecting pass names and local function calls."""

            def visit_Call(self, node):
                call_func = node.func

                # --- Pattern 1: xxx.transform.PassName(...) ---
                if isinstance(call_func, ast.Attribute):
                    names = []
                    cur = call_func
                    while isinstance(cur, ast.Attribute):
                        names.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        names.append(cur.id)
                    names.reverse()
                    if "transform" in names and len(names) > names.index("transform") + 1:
                        pass_name = names[names.index("transform") + 1]
                        if pass_name and pass_name[0].isupper():
                            passes.append(pass_name)

                # --- Pattern 2: LocalFunc(...) — follow into it ---
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


def _discover_passes(phase_func) -> list[str]:
    """Extract pass names from a phase function's source code via AST parsing.

    Looks for patterns like ``mod = xxx.transform.PassName(...)(mod)`` and
    extracts ``PassName`` in source order.  This enables pre-registering
    all passes as 'skipped' before execution, so the HTML report can show
    failed/skipped passes even when the phase crashes mid-way.

    Returns a list of pass display names (e.g. ``["Simplify", "InjectTmpBuffer"]``).
    """
    try:
        source = inspect.getsource(phase_func)
    except (OSError, TypeError):
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    passes = []

    class _PassVisitor(ast.NodeVisitor):
        """Walk the AST and collect pass constructor names.

        Matches call patterns like ``tilelang.transform.SomePass(...)`` or
        ``tir.transform.Simplify(...)`` — i.e. any attribute chain that
        contains a ``transform`` segment.  The final attribute is the pass name.
        """

        def visit_Call(self, node):
            func = node.func
            # Pattern: Attribute(value=..., attr=PassName)
            # where value chain contains a 'transform' segment
            if isinstance(func, ast.Attribute):
                names = []
                cur = func
                while isinstance(cur, ast.Attribute):
                    names.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    names.append(cur.id)
                names.reverse()
                if "transform" in names and len(names) > names.index("transform") + 1:
                    pass_name = names[names.index("transform") + 1]
                    # Pass classes are CamelCase (start with uppercase);
                    # filter out helpers like get_pass_context
                    if pass_name and pass_name[0].isupper():
                        passes.append(pass_name)
            self.generic_visit(node)

    _PassVisitor().visit(tree)
    return passes


def _traced_pipeline_lower(self, mod, target):
    """Intercept PassPipeline.lower to set phase context for pass tracing (new architecture)."""
    global _current_phase, _pass_index, _phase_call_count, _current_pass_index
    global _failed_pass_info, _records_offset, _auto_flush

    with _trace_lock:
        _phase_call_count += 1

        # Each compilation starts fresh
        reset()

        phase_name = f"pipeline_{self.name}"  # e.g. "pipeline_c", "pipeline_cuda"
        _current_phase = phase_name
        _pass_index = 0
        _current_pass_index = -1
        _failed_pass_info = None
        _records_offset = 0

        gen_html = _should_gen_html()

        # Pre-register all passes discovered via AST as "skipped"
        pass_names = _discover_passes_recursive(self._lower)
        if gen_html and pass_names:
            _ensure_trace_dir()
            for i, name in enumerate(pass_names):
                _records.append(
                    PassRecord(
                        phase=phase_name,
                        name=name,
                        index=i,
                        before_text="",
                        after_text="",
                        changed=False,
                        status=STATUS_SKIPPED,
                    )
                )

        _auto_flush = gen_html

    try:
        result = _original_pipeline_lower(self, mod, target)
    except Exception as e:
        _auto_flush = False
        _current_phase = None
        print(f"  [pass_trace] EXCEPTION in {phase_name}: {e}")

        # If the exception happened during pass factory creation
        # (before Pass.__call__), _traced_pass_call never ran for
        # this pass.  Mark the next expected pass as failed so
        # the HTML report distinguishes it from "skipped" passes.
        if _failed_pass_info is None and 0 <= _pass_index < len(_records):
            rec = _records[_pass_index]
            if rec.status == STATUS_SKIPPED:
                rec.status = STATUS_FAILED
                rec.error_msg = str(e)

        # Generate HTML even on failure (marks failed/skipped passes)
        if _records and _trace_dir:
            try:
                from .html import generate_html

                html_path = os.path.join(_trace_dir, "ir_trace.html")
                generate_html(_records, html_path)
                print(f"  [pass_trace] HTML report (with failures) written to: {html_path}")
            except Exception as html_err:
                print(f"  [pass_trace] WARNING: failed to generate HTML report: {html_err}")
                import traceback

                traceback.print_exc()

        raise

    _auto_flush = False
    _current_phase = None

    # Generate HTML report (single phase, so always generate)
    if _records and _trace_dir:
        from .html import generate_html

        html_path = os.path.join(_trace_dir, "ir_trace.html")
        generate_html(_records, html_path)
        print(f"  [pass_trace] HTML report written to: {html_path}")

    _phase_call_count = 0
    return result


def _discover_phases(lower_func) -> list:
    """Discover phase functions from the old architecture via bytecode scanning.

    Scans ``lower()`` for LOAD_GLOBAL instructions that reference functions
    from ``tilelang.engine.phase``, preserving source order.

    Returns a list of phase function objects.
    """
    try:
        from tilelang.engine import phase as phase_module
    except ImportError:
        return []

    # Scan bytecode of lower() for referenced global names
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

    # Fallback: all public functions from phase module, sorted by source line
    if not phase_funcs:
        phase_funcs = [
            getattr(phase_module, name)
            for name in sorted(dir(phase_module))
            if not name.startswith("_") and callable(getattr(phase_module, name, None))
        ]

    # Sort by source line to preserve execution order
    def _src_line(f):
        try:
            return inspect.getsourcelines(f)[1]
        except (OSError, TypeError):
            return 999999

    phase_funcs.sort(key=_src_line)
    return phase_funcs


def patch():
    """Activate IR pass tracing via monkey-patching.

    Supports two architectures:

    **New (PassPipeline)**: each backend registers a ``PassPipeline`` object,
    dispatched through ``PassPipeline.lower(mod, target)``.  We patch this
    single method to intercept all backends.

    **Old (phase-based)**: ``lower()`` calls phase functions from
    ``tilelang.engine.phase`` (e.g. ``LowerAndLegalize``, ``OptimizeForTarget``).
    We discover them via bytecode scanning and wrap each individually.

    Both share the same ``Pass.__call__`` interception for per-pass IR capture.
    Architecture is auto-detected at patch time.
    """
    from tvm.ir.transform import Pass

    # 1. Patch Pass.__call__ — intercept ALL pass executions (guard against double-patch)
    global _original_pass_call, _num_phases, _original_pipeline_lower
    if _original_pass_call is None:
        _original_pass_call = Pass.__call__
        Pass.__call__ = _traced_pass_call

    # 2. Already patched phase/pipeline level? Skip.
    if _original_pipeline_lower is not None:
        return

    # 3. Try new architecture: PassPipeline
    try:
        from tilelang.backend.pass_pipeline import PassPipeline

        _original_pipeline_lower = PassPipeline.lower
        PassPipeline.lower = _traced_pipeline_lower
        _num_phases = 1
        print(
            "[pass_trace] IR pass tracing patched (PassPipeline architecture). "
            "Set TILELANG_PASS_TRACE=1 to enable."
        )
        return
    except ImportError:
        pass

    # 4. Fallback: old phase-based architecture
    try:
        import tilelang.engine.lower as lower_mod

        lower_func = lower_mod.lower
        patch_mod = lower_mod
    except (ImportError, AttributeError):
        try:
            from tilelang.engine import lower as lower_func

            import tilelang.engine as patch_mod
        except (ImportError, AttributeError) as e:
            print(f"[pass_trace] WARNING: could not patch — {e}")
            return

    phase_funcs = _discover_phases(lower_func)
    for i, phase_func in enumerate(phase_funcs):
        wrapped = _wrap_phase(phase_func, i + 1, len(phase_funcs))
        # Patch on patch_mod (handles `import tilelang.engine.phase as phase; phase.LowerAndLegalize(...)`)
        setattr(patch_mod, phase_func.__name__, wrapped)
        # Also patch on the phase module itself (handles `phase_module.FuncName(...)` from any caller)
        try:
            from tilelang.engine import phase as phase_module

            if hasattr(phase_module, phase_func.__name__):
                setattr(phase_module, phase_func.__name__, wrapped)
        except ImportError:
            pass
        # Also patch lower_func's globals (handles `from tilelang.engine.phase import LowerAndLegalize`)
        # This is the most common pattern — from-imports create local name bindings at import time.
        if phase_func.__name__ in getattr(lower_func, "__globals__", {}):
            lower_func.__globals__[phase_func.__name__] = wrapped
    _num_phases = len(phase_funcs)
    # Mark as patched (use a sentinel since _original_pipeline_lower is unused in old arch)
    _original_pipeline_lower = True
    print(
        f"[pass_trace] IR pass tracing patched (phase-based architecture, "
        f"{_num_phases} phases). Set TILELANG_PASS_TRACE=1 to enable."
    )


def reset():
    """Clear collected records and cached trace dir (useful between multiple compilations).

    Note: does NOT reset _phase_call_count — that tracks compilation boundaries
    and is managed by the phase wrappers.
    """
    global _records, _trace_dir, _current_phase, _pass_index, _current_pass_index, _failed_pass_info, _records_offset, _auto_flush
    _records = []
    _trace_dir = None
    _current_phase = None
    _pass_index = 0
    _current_pass_index = -1
    _failed_pass_info = None
    _records_offset = 0
    _auto_flush = False
