import dace
from dace.properties import make_properties
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation
import typing

@transformation.explicit_cf_compatible
@make_properties
class DoubleBuffering(transformation.SingleStateTransformation):
    device_map_type = dace.properties.Property(dtype=dace.dtypes.ScheduleType,
                                               default=dace.dtypes.ScheduleType.GPU_Device,
                                               desc="The schedule type for the device map.",
                                               allow_none=True)
    copy_src_type = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Global,
        desc="The storage type for the source of the double buffering copy.",
        allow_none=True,
    )
    copy_dst_type = dace.properties.Property(
        dtype=dace.dtypes.StorageType,
        default=dace.dtypes.StorageType.GPU_Shared,
        desc="The storage type for the destination of the double buffering copy.",
        allow_none=True,
    )
    map_entry = transformation.PatternNode(dace.nodes.MapEntry)

    def __init__(self,
                 device_map_type: dace.dtypes.ScheduleType = None,
                 copy_src_type: dace.dtypes.StorageType = None,
                 copy_dst_type: dace.dtypes.StorageType = None,
                 **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.device_map_type = device_map_type
        self.copy_src_type = copy_src_type
        self.copy_dst_type = copy_dst_type

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

        if self.device_map_type is None or self.copy_src_type is None or self.copy_dst_type is None:
            return False

        # For now only GPU codegen supports this
        if self.device_map_type != dace.dtypes.ScheduleType.GPU_Device:
            #    raise ValueError("The device_map_type must be set to GPU_Device for this transformation currently. TO-DO")
            return False
        if self.copy_src_type != dace.dtypes.StorageType.GPU_Global:
            #    raise ValueError("The copy_src_type must be set to GPU_Global for this transformation currently. TO-DO")
            return False
        if self.copy_dst_type != dace.dtypes.StorageType.GPU_Shared:
            #    raise ValueError("The copy_dst_type must be set to GPU_Shared for this transformation currently. TO-DO")
            return False

        # At least one copy from src to dst within the kernel
        kernel_nodes = graph.all_nodes_between(self.map_entry, graph.exit_node(self.map_entry))
        kernel_edges = graph.all_edges(*kernel_nodes)
        has_src_to_dst_copy = False
        for edge in kernel_edges:
            if isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode):
                src_arr = sdfg.arrays[edge.src.data]
                dst_arr = sdfg.arrays[edge.dst.data]
                if (src_arr.storage == self.copy_src_type and
                    dst_arr.storage == self.copy_dst_type):
                    has_src_to_dst_copy = True
                    break
        if not has_src_to_dst_copy:
            return False

        return True

    def apply(self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        return
