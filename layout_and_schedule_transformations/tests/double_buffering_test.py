import copy
import numpy as np
import dace
import pytest

from layout_and_schedule_transformations.double_buffering import DoubleBuffering
from layout_and_schedule_transformations.tests.test_utils import _add_shared_memory

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
        for i in dace.map[0:N:512] @ dace.dtypes.ScheduleType.GPU_Device:
            for k in dace.map[0:2] @ dace.dtypes.ScheduleType.Sequential:
                for j in dace.map[0:256] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                    C[i + j + k * 256] = A[i + j + k * 256] + B[i + j + k * 256]

    # Create original SDFG
    original_sdfg = kernel.to_sdfg(use_cache=False, simplify=False)
    original_sdfg.simplify()
    original_sdfg.save("original_sdfg.sdfg")

    original_sdfg.validate()

    # Create transformed SDFG
    transformed_sdfg = copy.deepcopy(original_sdfg)
    transformed_sdfg.name = original_sdfg.name + "_double_buffered"
    _add_shared_memory(transformed_sdfg, add_src_access_node=False)
    transformed_sdfg.save("original_sdfg_with_shared_memory.sdfg")

    # Apply transformations
    for state in transformed_sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                # Apply double buffering transformation
                options_dict = {
                    "device_map_type":dace.dtypes.ScheduleType.GPU_Device,
                    "copy_src_type":dace.dtypes.StorageType.GPU_Global,
                    "copy_dst_type":dace.dtypes.StorageType.GPU_Shared,
                }
                db_transform_can_be_applied = DoubleBuffering(**options_dict).can_be_applied_to(
                    sdfg=transformed_sdfg,
                    options=options_dict,
                    map_entry=node,
                )
                assert db_transform_can_be_applied, f"DoubleBuffering transformation should be applicable to the map entry. Returned:{db_transform_can_be_applied}"

                DoubleBuffering(**options_dict).apply_to(
                    map_entry=node,
                    sdfg=transformed_sdfg
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
    exit(0 if success else 1)