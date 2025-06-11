import dace
from dace.sdfg import utils as sdutil
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation

@transformation.explicit_cf_compatible
class EmptyTransformation(transformation.MultiStateTransformation):
    @staticmethod
    def annotates_memlets():
        return False

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph: ControlFlowRegion, expr_index, sdfg: dace.SDFG, permissive=False):
        return True

    def apply(self, graph: ControlFlowRegion, sdfg: dace.SDFG):
        return
