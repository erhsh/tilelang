"""Source-only lowering establishes and restores its resolved target scope."""

import pytest
import tilelang
from tilelang.ascend import language as T
from tvm.target import Target


@T.prim_func
def vector_copy(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
    with T.Kernel(1), T.SimtVF(threads=32):
        for i in T.Parallel(32):
            for j in T.vectorized(4):
                B[i * 4 + j] = A[i * 4 + j] + 1


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("outer_target", [None, "c"])
def test_source_only_lowering_owns_target_context(target, outer_target):
    before = Target.current()
    if outer_target is None:
        artifact = tilelang.lower(vector_copy, target=target)
    else:
        outer = Target(outer_target)
        with outer:
            artifact = tilelang.lower(vector_copy, target=target)
            assert Target.current().same_as(outer)
    assert artifact.kernel_source
    assert ("with pto.simt(" if target == "pto" else "__simt_vf__") in artifact.kernel_source
    after = Target.current()
    assert after is None if before is None else after.same_as(before)
