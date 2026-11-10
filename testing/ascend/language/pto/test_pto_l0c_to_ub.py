"""Rectangular NZ accumulators use an M-panel source stride when draining."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.backend import create_backend_context

from testing.ascend._ir import calls

pytestmark = pytest.mark.pto


def rectangular_copy(m, n):
    @T.prim_func
    def kernel(A: T.Tensor((m, 16), "bfloat16"), B: T.Tensor((n, 16), "bfloat16"), C: T.Tensor((m, n), "bfloat16")):
        with T.MixedKernel(1, sids=1):
            a = T.alloc_l1((m, 16), "bfloat16")
            b = T.alloc_l1((n, 16), "bfloat16")
            accumulator = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, accumulator, transpose_B=True, clear_accum=True)
            T.copy(accumulator, ub, sub_blockid=0)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(m, n):
                    C[i, j] = T.Cast("bfloat16", ub[i, j])

    return kernel


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32)], ids=["wide", "tall"])
def test_pto_rectangular_nz_source_stride(m, n):
    context = create_backend_context("pto")
    # This regression must also work without the separate source-target fix.
    with context.target:
        artifact = tilelang.lower(rectangular_copy(m, n), target=context.target)
    drains = [call for func in artifact.device_mod.functions.values() for call in calls(func, "tl.ascend_copy_matrix_cc_to_ub")]
    assert len(drains) == 1
    assert [int(value) for value in drains[0].args[3:7]] == [n, m, n, m]
    mte = next(line for line in artifact.kernel_source.splitlines() if "pto.mte_l0c_ub(" in line)
    assert f', {m}, {n}, {m}, {n}, 0, layout="nz2nd")' in mte


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("m,n", [(32, 64), (64, 32)], ids=["wide", "tall"])
def test_pto_rectangular_nz_copy_numerics(m, n):
    kernel = tilelang.compile(rectangular_copy(m, n), target="pto", execution_backend="pto", out_idx=-1)
    a = torch.randn(m, 16, dtype=torch.bfloat16, device="npu") * 0.125
    b = torch.randn(n, 16, dtype=torch.bfloat16, device="npu") * 0.125
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    torch.testing.assert_close(kernel(a, b), expected, rtol=2e-2, atol=2e-2)
