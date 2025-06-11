import copy
import numpy as np
import dace


import DaceLayoutAndScheduleTransformations
from DaceLayoutAndScheduleTransformations import DoubleBuffering

def test_standalone_execution():
    """Standalone test function that can be run without pytest."""
    print("Running standalone permute transformations test...")

    # Setup
    dace.Config.set('cache', value='unique')

    # Create kernel
    N = dace.symbol("N", dtype=dace.int64)
    N_val = 1024

    @dace.program
    def kernel(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N] @ dace.dtypes.ScheduleType.GPU_Device:
            C[i] = A[i] + B[i]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()

    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = original_sdfg.name + "_double_buffered"

    # Apply transformations
    for state in transformed_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                # Apply double buffering transformation
                DoubleBuffering.DoubleBuffering(
                    device_map_type=dace.dtypes.ScheduleType.GPU_Device,
                    double_buffering_copy_src=dace.dtypes.StorageType.GPU_Global,
                    double_buffering_copy_dst=dace.dtypes.StorageType.GPU_Shared,
                ).apply_to(
                    map_entry=node,
                    sdfg=transformed_sdfg,
                    cfg=state,
                )

    # Validate SDFGs
    original_sdfg.validate()
    transformed_sdfg.validate()

    # Initialize data
    np.random.seed(42)
    vals_A_orig = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N_val, (N_val,), dtype=np.float64)
    vals_B_orig = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N_val, (N_val,), dtype=np.float64)
    vals_C_orig = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N_val, (N_val,), dtype=np.float64)

    vals_A_2 = vals_A_orig.copy()
    vals_B_2= vals_B_orig.copy()
    vals_C_2 = vals_C_orig.copy()

    # Execute SDFGs
    original_sdfg(vals_A=vals_A_orig, vals_B=vals_B_orig, vals_C=vals_C_orig, N=N_val)
    transformed_sdfg(vals_A=vals_A_2, vals_B=vals_B_2, vals_C=vals_C_2, N=N_val)

    # Check results
    vals_A_close = np.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)
    vals_B_close = np.allclose(vals_C_orig, vals_C_2, rtol=1e-10, atol=1e-12)

    print(f"vals_A results match: {vals_A_close}")
    print(f"vals_B results match: {vals_B_close}")

    if vals_A_close and vals_B_close:
        print("✅ All tests passed! Permute transformations preserve correctness.")
    else:
        print("❌ Test failed! Results differ between original and transformed SDFGs.")
        if not vals_A_close:
            print(f"vals_A max difference: {np.max(np.abs(vals_A_orig - vals_A_2))}")
            print(f"vals_A difference: {np.abs(vals_A_orig - vals_A_2)}")
        if not vals_B_close:
            print(f"vals_B max difference: {np.max(np.abs(vals_B_orig - vals_B_2))}")
            print(f"vals_B difference: {np.abs(vals_B_orig - vals_B_2)}")

    return vals_A_close and vals_B_close


if __name__ == "__main__":
    # Run standalone test
    success = test_standalone_execution()
    exit(0 if success else 1)