import dace
from dace.properties import make_properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation


@transformation.explicit_cf_compatible
@make_properties
class EmptyTransformation(transformation.MultiStateTransformation):
    device_map_type = dace.properties.Property(dtype=dace.dtypes.ScheduleType,
                                               default=dace.dtypes.ScheduleType.GPU_Device,
                                               desc="The schedule type for the device map.")
    double_buffering_copy_src = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Global,
        desc="The storage type for the source of the double buffering copy."
    )
    double_buffering_copy_dst = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Shared,
        desc="The storage type for the destination of the double buffering copy."
    )


    def __init__(self, device_map_type: dace.dtypes.ScheduleType):
        self.device_map_type = device_map_type
        if self.device_map_type != dace.dtypes.ScheduleType.GPU_Device:
            raise ValueError("The device_map_type must be set to GPU_Device for this transformation currently. TO-DO")
        if self.double_buffering_copy_src != dace.dtypes.StorageType.GPU_Global:
            raise ValueError("The double_buffering_copy_src must be set to GPU_Global for this transformation currently. TO-DO")
        if self.double_buffering_copy_dst != dace.dtypes.StorageType.GPU_Shared:
            raise ValueError("The double_buffering_copy_dst must be set to GPU_Shared for this transformation currently. TO-DO")

    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index, sdfg: dace.SDFG, permissive=False):
        # Is map entry
        if not isinstance(self.map_entry, dace.sdfg.nodes.MapEntry):
            return False

        # Ensure device map
        if self.map_entry.map.schedule != self.device_map_type:
            return False

        # Kernel directly in a state
        if not isinstance(graph, dace.SDFGState):
            return False

        # At least one copy from src to dst within the kernel
        kernel_nodes = graph.all_nodes_between(self.map_entry, graph.exit_node(self.map_entry))
        kernel_edges = graph.all_edges(*kernel_nodes)
        has_src_to_dst_copy = False
        for edge in kernel_edges:
            if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode):
                if (edge.src.data.storage == self.double_buffering_copy_src and
                        edge.dst.data.storage == self.double_buffering_copy_dst):
                    has_src_to_dst_copy = True
                    break
        if not has_src_to_dst_copy:
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        return
