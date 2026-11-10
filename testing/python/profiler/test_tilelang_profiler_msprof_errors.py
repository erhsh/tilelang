"""Missing profiler records are distinct from errors in workload execution."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tilelang.profiler import msprof
from tilelang.profiler.errors import ProfileUnavailableError


@pytest.fixture
def empty_msprof(monkeypatch, tmp_path):
    profiler = ModuleType("torch_npu.profiler")

    class Trace:
        prof_if = SimpleNamespace(prof_path=str(tmp_path))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def step(self):
            pass

    profiler.profile = lambda **kwargs: Trace()
    profiler.schedule = lambda **kwargs: None
    profiler.ProfilerActivity = SimpleNamespace(NPU=0)
    profiler.ProfilerLevel = SimpleNamespace(Level1=0)
    profiler.AiCMetrics = SimpleNamespace(PipeUtilization=0)
    profiler._ExperimentalConfig = lambda **kwargs: None
    npu = ModuleType("torch_npu")
    npu.__path__ = []
    npu.profiler = profiler
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", profiler)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    monkeypatch.setattr(msprof, "get_msprof_cache_flush_kernel", lambda size: lambda cache: None)
    return torch.empty(1, dtype=torch.int32)


def test_msprof_empty_records_have_a_distinct_error(empty_msprof):
    with pytest.raises(ProfileUnavailableError, match="No profiled kernels"):
        msprof.bench_with_msprof(lambda: None, empty_msprof, n_repeat=1)


def test_msprof_does_not_reclassify_workload_errors(empty_msprof):
    failure = RuntimeError("device execution failed")

    def workload():
        raise failure

    with pytest.raises(RuntimeError) as result:
        msprof.bench_with_msprof(workload, empty_msprof, n_repeat=1)
    assert result.value is failure
