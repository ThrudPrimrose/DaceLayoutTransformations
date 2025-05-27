import dace
from typing import Dict, List, Any

from dace import properties
from dace.transformation.dataflow.map_dim_shuffle import MapDimShuffle
from dace.transformation import pass_pipeline as ppl

from dataclasses import dataclass


@dataclass
class PermuteMapDimensions(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes & ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def __init__(self,
                 permute_map: Dict[dace.nodes.MapEntry, List[int]] | Dict[str, List[int]],
                 use_labels: bool):
        if use_labels:
            self._permute_map_label = permute_map
        else:
            self._permute_map_node = permute_map
        self._use_labels = use_labels

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: Dict[str, Any]) -> int:
        if self._use_labels:
            self._permute_map_dimensions_from_label(sdfg, self._permute_map_label)
        else:
            self._permute_map_dimensions(sdfg, self._permute_map_node)

        return 0


    def _permute_map_dimensions_from_label(self, sdfg: dace.SDFG, permute_map : Dict[dace.nodes.MapEntry, List[int]]):
        permute_map_from_nodes = dict()
        for state, _ in sdfg.all_nodes_recursive():
            if isinstance(state, dace.SDFGState):
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        if node.map.label in permute_map:
                            permute_map_from_nodes[node] = permute_map[node.map.label]
        self._permute_map_dimensions(sdfg, permute_map_from_nodes)

    def _permute_map_dimensions(self, sdfg: dace.SDFG, permute_map : Dict[str, List[int]]):
        for state, _ in sdfg.all_nodes_recursive():
            if isinstance(state, dace.SDFGState):
                for node in state.nodes():
                    if isinstance(node, dace.nodes.MapEntry):
                        old_params = node.map.params
                        new_params = []
                        if node.map.label in permute_map:
                            for j in range(len(permute_map[node.map.label])):
                                new_params.append(old_params[permute_map[node.map.label][j]])
                            MapDimShuffle.apply_to(sdfg, map_entry=node, options={"parameters": new_params})