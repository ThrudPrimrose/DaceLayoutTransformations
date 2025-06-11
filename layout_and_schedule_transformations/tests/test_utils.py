import dace
import copy

def _add_shared_memory(sdfg: dace.SDFG, add_src_access_node: bool = False):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                next_map = None
                for n in state.bfs_nodes(node):
                    if isinstance(n, dace.sdfg.nodes.MapEntry) and n != node:
                        next_map = n
                        break
                    elif isinstance(n, dace.nodes.MapExit):
                        break
                if next_map is None:
                    raise ValueError("No next map found for the GPU_Device map entry.")

                src_name_dst_name_offset = dict()
                edges_to_rm = set()
                for in_edge in state.in_edges(next_map):
                    if in_edge.data is not None:
                        in_arr_name = in_edge.data.data
                        copy_shape = [(0, (((e) - b)//s), 1) for b, e, s in in_edge.data.subset]
                        copied_shape = [(((e + 1) - b)//s) for b, e, s in in_edge.data.subset]
                        copy_offset = [b for b, _, _ in in_edge.data.subset]
                        shared_mem_name = "shr_" + in_arr_name
                        in_arr = sdfg.arrays[in_arr_name]
                        if shared_mem_name not in sdfg.arrays:
                            sdfg.add_array(shared_mem_name, copied_shape, in_arr.dtype, storage=dace.dtypes.StorageType.GPU_Shared, transient=True)

                        if add_src_access_node is True:
                            a1 = state.add_access(in_arr_name)
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(a1, None, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=in_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name,
                                                                    sdfg.arrays[shared_mem_name]))
                            e3 = state.add_edge(in_edge.src, in_edge.src_conn, a1, None,
                                                copy.deepcopy(in_edge.data))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)
                        else:
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(in_edge.src, in_edge.src_conn, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=in_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, next_map, in_edge.dst_conn,
                                                dace.Memlet.from_array(shared_mem_name,
                                                                    sdfg.arrays[shared_mem_name]))
                            edges_to_rm.add(in_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_offset)

                nodes = state.all_nodes_between(next_map, state.exit_node(next_map))
                for edge in state.all_edges(*nodes):
                    if edge.data is not None and edge.data.data in src_name_dst_name_offset:
                        dst_name, offset = src_name_dst_name_offset[edge.data.data]
                        edge.data.data = dst_name
                        old_subset = [(b,e,s) for b, e, s in edge.data.subset]
                        new_subset = [(b - offset[i], e - offset[i], s) for i, (b, e, s) in enumerate(old_subset)]
                        edge.data.subset = dace.subsets.Range(new_subset)

                for edge in edges_to_rm:
                    state.remove_edge(edge)


def _add_shared_memory_no_tblock_map(sdfg: dace.SDFG, add_src_access_node: bool = False, tblock_size: int = 32):
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.MapEntry) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                next_map = None
                for n in state.bfs_nodes(node):
                    if isinstance(n, dace.sdfg.nodes.MapEntry) and n != node:
                        next_map = n
                        break
                    elif isinstance(n, dace.nodes.MapExit):
                        break
                assert next_map is None

                src_name_dst_name_offset = dict()
                edges_to_rm = set()
                for out_edge in state.out_edges(node):
                    if out_edge.data is not None:
                        in_arr_name = out_edge.data.data
                        copy_shape = [(dace.symbolic.SymExpr(f"{b}%{tblock_size}"), dace.symbolic.SymExpr(f"{b}%{tblock_size}"), 1) for b, e, s in out_edge.data.subset]
                        copied_shape = [(((e + 1) - b)//s)*tblock_size for b, e, s in out_edge.data.subset]
                        copy_offset = [b for b, _, _ in out_edge.data.subset]
                        shared_mem_name = "shr_" + in_arr_name
                        in_arr = sdfg.arrays[in_arr_name]
                        if shared_mem_name not in sdfg.arrays:
                            sdfg.add_array(shared_mem_name, copied_shape, in_arr.dtype, storage=dace.dtypes.StorageType.GPU_Shared, transient=True)

                        if add_src_access_node is True:
                            a1 = state.add_access(in_arr_name)
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(a1, None, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=out_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, out_edge.dst, out_edge.dst_conn,
                                                dace.Memlet(data=shared_mem_name,
                                                            subset=dace.subsets.Range(copy_shape)))
                            e3 = state.add_edge(out_edge.src, out_edge.src_conn, a1, None,
                                                copy.deepcopy(out_edge.data))
                            edges_to_rm.add(out_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_shape)
                        else:
                            a2 = state.add_access(shared_mem_name)
                            e1 = state.add_edge(out_edge.src, out_edge.src_conn, a2, None, dace.Memlet(
                                data=in_arr_name,
                                subset=out_edge.data.subset,
                                other_subset=dace.subsets.Range(copy_shape),
                                wcr=None,
                            ))
                            e2 = state.add_edge(a2, None, out_edge.dst, out_edge.dst_conn,
                                                dace.Memlet(data=shared_mem_name,
                                                            subset=dace.subsets.Range(copy_shape)))
                            edges_to_rm.add(out_edge)
                            src_name_dst_name_offset[in_arr_name] = (shared_mem_name, copy_shape)

                nodes = state.all_nodes_between(node, state.exit_node(node))
                for edge in state.all_edges(*nodes):
                    if isinstance(edge.dst, dace.nodes.AccessNode) and edge.data.data in src_name_dst_name_offset:
                        continue
                    if edge.data is not None and edge.data.data in src_name_dst_name_offset:
                        dst_name, new_subset = src_name_dst_name_offset[edge.data.data]
                        edge.data.data = dst_name
                        edge.data.subset = dace.subsets.Range(new_subset)
                        edge.data.src_subset = dace.subsets.Range(new_subset)
                        edge.data.dst_subset = dace.subsets.Range(new_subset)
                        edge.data.other_subset = dace.subsets.Range(new_subset)

                for edge in edges_to_rm:
                    state.remove_edge(edge)
