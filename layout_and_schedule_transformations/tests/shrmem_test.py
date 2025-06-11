import copy
import dace
import pytest
import cupy as cp

from layout_and_schedule_transformations.tests.test_utils import _add_shared_memory, _add_shared_memory_no_tblock_map

def test_standalone_execution():
    """Standalone test function that can be run without pytest."""
    print("Running standalone Shared Memory transformations test...")

    # Setup
    dace.Config.set('cache', value='unique')

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)
    N_val = 1024
    K = 1

    @dace.program
    def kernel(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N:256*K] @ dace.dtypes.ScheduleType.GPU_Device:
            for k in dace.map[0:K] @ dace.dtypes.ScheduleType.Sequential:
                for j in dace.map[0:256] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                    C[i + j + k * 256] = A[i + j + k * 256] + B[i + j + k * 256]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()
    original_sdfg.save("original_sdfg.sdfg")


    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    _add_shared_memory(transformed_sdfg, add_src_access_node=False)
    transformed_sdfg.save("transformed_sdfg_with_shared_memory.sdfg")
    transformed_sdfg.validate()

    # Validate SDFGs
    original_sdfg.validate()
    transformed_sdfg.validate()

    # Initialize data
    cp.random.seed(42)
    vals_A_orig = cp.fromfunction(lambda i,: (i * 2) / N_val, (N_val,), dtype=cp.float64)
    vals_B_orig = cp.fromfunction(lambda i,: (i * 3) / N_val, (N_val,), dtype=cp.float64)
    vals_C_orig = cp.fromfunction(lambda i,: (i * 5) / N_val, (N_val,), dtype=cp.float64)

    vals_A_2 = vals_A_orig.copy()
    vals_B_2= vals_B_orig.copy()
    vals_C_2 = vals_C_orig.copy()

    # Execute SDFGs
    original_sdfg(A=vals_A_orig, B=vals_B_orig, C=vals_C_orig, N=N_val)
    transformed_sdfg(A=vals_A_2, B=vals_B_2, C=vals_C_2, N=N_val)

    # Check results
    vals_A_close = cp.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)
    vals_B_close = cp.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)

    print(f"vals_A results match: {vals_A_close}")
    print(f"vals_B results match: {vals_B_close}")

    if vals_A_close and vals_B_close:
        print("Shared Memory transformations preserve correctness.")
    else:
        print("Results differ between original and transformed SDFGs.")
        if not vals_A_close:
            print(f"vals_A max difference: {cp.max(cp.abs(vals_A_orig - vals_A_2))}")
            print(f"vals_A difference: {cp.abs(vals_A_orig - vals_A_2)}")
        if not vals_B_close:
            print(f"vals_B max difference: {cp.max(cp.abs(vals_B_orig - vals_B_2))}")
            print(f"vals_B difference: {cp.abs(vals_B_orig - vals_B_2)}")

    assert vals_A_close and vals_B_close
    return vals_A_close and vals_B_close


def test_standalone_execution_2():
    """Standalone test function that can be run without pytest."""
    print("Running standalone Shared Memory transformations test...")

    # Setup
    dace.Config.set('cache', value='unique')

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)
    N_val = 1024
    K = 1

    @dace.program
    def kernel(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N:1] @ dace.dtypes.ScheduleType.GPU_Device:
                    C[i] = A[i] + B[i]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()
    original_sdfg.save("original_sdfg.sdfg")


    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    _add_shared_memory_no_tblock_map(transformed_sdfg, add_src_access_node=False)
    transformed_sdfg.save("transformed_sdfg_with_shared_memory.sdfg")
    transformed_sdfg.validate()

    # Validate SDFGs
    original_sdfg.validate()
    transformed_sdfg.validate()

    # Initialize data
    cp.random.seed(42)
    vals_A_orig = cp.fromfunction(lambda i,: (i * 2) / N_val, (N_val,), dtype=cp.float64)
    vals_B_orig = cp.fromfunction(lambda i,: (i * 3) / N_val, (N_val,), dtype=cp.float64)
    vals_C_orig = cp.fromfunction(lambda i,: (i * 5) / N_val, (N_val,), dtype=cp.float64)

    vals_A_2 = vals_A_orig.copy()
    vals_B_2= vals_B_orig.copy()
    vals_C_2 = vals_C_orig.copy()

    # Execute SDFGs
    original_sdfg(A=vals_A_orig, B=vals_B_orig, C=vals_C_orig, N=N_val)
    transformed_sdfg(A=vals_A_2, B=vals_B_2, C=vals_C_2, N=N_val)

    # Check results
    vals_C_close = cp.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)

    print(f"vals_C results match: {vals_C_close}")

    if vals_C_close:
        print("Test Fail: Shared Memory transformations preserve correctness, but they should not be synchronized by the current codegen.")
    else:
        print("Test Pass: Weird Shared Memory transformations can't be synchronized by the current codegen.")
        if not vals_C_close:
            print(f"vals_A max difference: {cp.max(cp.abs(vals_C_orig - vals_C_2))}")
            print(f"vals_A difference: {cp.abs(vals_C_orig - vals_C_2)}")
    assert not vals_C_close

if __name__ == "__main__":
    success = test_standalone_execution()
    success = success and test_standalone_execution_2()
    exit(0 if success else 1)