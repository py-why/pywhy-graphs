from typing import List, Optional, Tuple

import numpy as np

import pywhy_graphs
from pywhy_graphs.classes.functions import edge_types
from pywhy_graphs.config import CLearnEndpoint, EdgeType
from pywhy_graphs.typing import Node


def graph_to_clearn(G) -> Tuple[np.ndarray, List[Node]]:
    # define the array
    arr = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=int)

    # get the array index based on the order of nodes inside graph
    arr_idx = list(G.nodes)

    for u in G.nodes:
        for v in G.nodes:
            # if the nodes are the same, skip
            if u == v:
                continue

            # if there is no adjacency among u and v, then skip
            if v not in G.neighbors(u):
                continue

            # get indices
            udx = arr_idx.index(u)
            vdx = arr_idx.index(v)

            # at this point, there is an edge among u and v
            uv_edge_types = edge_types(G, u, v)
            if len(uv_edge_types) == 1:
                edge_type = uv_edge_types[0]
                # if there is only one edge type and it is directed, then
                # there are
                if edge_type == EdgeType.DIRECTED.value:
                    if G.has_edge(u, v, EdgeType.DIRECTED.value):
                        # u -> v
                        endpoint_v = CLearnEndpoint.ARROW
                        endpoint_u = CLearnEndpoint.TAIL
                    else:
                        # u <- v
                        endpoint_u = CLearnEndpoint.ARROW
                        endpoint_v = CLearnEndpoint.TAIL
                elif edge_type == EdgeType.BIDIRECTED.value:
                    # <->
                    endpoint_v = CLearnEndpoint.ARROW
                    endpoint_u = CLearnEndpoint.ARROW
                elif edge_type == EdgeType.UNDIRECTED.value:
                    # --
                    endpoint_v = CLearnEndpoint.TAIL
                    endpoint_u = CLearnEndpoint.TAIL
                elif edge_type == EdgeType.CIRCLE.value:
                    if G.has_edge(u, v, EdgeType.CIRCLE.value) and G.has_edge(
                        v, u, EdgeType.CIRCLE.value
                    ):
                        # u o-o v
                        endpoint_v = CLearnEndpoint.CIRCLE
                        endpoint_u = CLearnEndpoint.CIRCLE
                    elif G.has_edge(u, v, EdgeType.CIRCLE.value) and not G.has_edge(v, u):
                        # u --o v
                        endpoint_v = CLearnEndpoint.CIRCLE
                        endpoint_u = CLearnEndpoint.TAIL
                    elif not G.has_edge(u, v) and G.has_edge(v, u, EdgeType.CIRCLE.value):
                        # u o-- v
                        endpoint_v = CLearnEndpoint.TAIL
                        endpoint_u = CLearnEndpoint.CIRCLE
                else:
                    raise RuntimeError(
                        f"Unrecognizd edge type {edge_type}. Use one of "
                        f"{[edge.value for edge in EdgeType]}."
                    )
            elif len(uv_edge_types) == 2:
                if (EdgeType.DIRECTED.value in uv_edge_types) and (
                    EdgeType.BIDIRECTED.value in uv_edge_types
                ):
                    if G.has_edge(u, v, EdgeType.DIRECTED.value):
                        # u -> v and u <-> v
                        endpoint_v = CLearnEndpoint.ARROW_AND_ARROW
                        endpoint_u = CLearnEndpoint.TAIL_AND_ARROW
                    else:
                        # u <- v and u <-> v
                        endpoint_u = CLearnEndpoint.ARROW_AND_ARROW
                        endpoint_v = CLearnEndpoint.TAIL_AND_ARROW
                elif (EdgeType.DIRECTED.value in uv_edge_types) and (
                    EdgeType.UNDIRECTED.value in uv_edge_types
                ):
                    if G.has_edge(u, v, EdgeType.DIRECTED.value):
                        # u -> v and u -- v
                        endpoint_v = CLearnEndpoint.TAIL_AND_ARROW
                        endpoint_u = CLearnEndpoint.TAIL_AND_TAIL
                    else:
                        # u <- v and u -- v
                        endpoint_u = CLearnEndpoint.TAIL_AND_ARROW
                        endpoint_v = CLearnEndpoint.TAIL_AND_TAIL
                elif (EdgeType.BIDIRECTED.value in uv_edge_types) and (
                    EdgeType.UNDIRECTED.value in uv_edge_types
                ):
                    # u -- v and u <-> v
                    endpoint_v = CLearnEndpoint.TAIL_AND_ARROW
                    endpoint_u = CLearnEndpoint.TAIL_AND_ARROW
                elif EdgeType.CIRCLE.value in uv_edge_types:
                    # u *-o v or u o-* v
                    if G.has_edge(u, v, EdgeType.CIRCLE.value) and G.has_edge(
                        v, u, EdgeType.DIRECTED.value
                    ):
                        # u <-o v
                        endpoint_v = CLearnEndpoint.CIRCLE
                        endpoint_u = CLearnEndpoint.ARROW
                    elif G.has_edge(v, u, EdgeType.CIRCLE.value) and G.has_edge(
                        u, v, EdgeType.DIRECTED.value
                    ):
                        # u o-> v
                        endpoint_u = CLearnEndpoint.CIRCLE
                        endpoint_v = CLearnEndpoint.ARROW
            else:
                raise RuntimeError(
                    f"Causal-learn does not support more than two types of edges between nodes. "
                    f"There are {len(uv_edge_types)} edge types between {u} and {v}."
                )

            # set the array to the endpoint values
            arr[udx, vdx] = endpoint_u.value
            arr[vdx, udx] = endpoint_v.value

    return arr, arr_idx


def clearn_to_graph(arr: np.ndarray, arr_idx: List[Node], graph_type: str):
    """Convert causal-learn array to a graph object.

    Parameters
    ----------
    arr : np.ndarray of shape (n_nodes, n_nodes)
        The causal-learn array encoding the endpoints between nodes.
        Columns are the starting node and rows are the ending node.
    arr_idx : List[Node] of length (n_nodes)
        The array index, which stores the name of the n_nodes in order of their
        rows/columns in ``arr``.
    graph_type : str, optional
        The type of causal graph. Must be one of 'dag', 'admg', 'cpdag', 'pag'.

    Returns
    -------
    graph : pywhy_nx.MixedEdgeGraph
        The causal graph.
    """
    if arr.shape[0] != arr.shape[1]:
        raise RuntimeError("Only square arrays are convertible to pywhy-graphs.")

    n_nodes = arr.shape[0]
    if len(arr_idx) != n_nodes:
        raise RuntimeError(
            f"The number of node names in order of the array rows/columns, {len(arr_idx)} "
            f"should match the number of rows/columns in array, {n_nodes}."
        )

    unique_edge_nums = np.unique(arr)
    try:
        any(CLearnEndpoint(num) not in CLearnEndpoint for num in unique_edge_nums)
    except ValueError as e:
        raise RuntimeError(
            f"Some entries of array are not causal-learn specified, specifically: {e}"
        )

    # TODO: enable us to infer the type?
    # instantiate the type of causal graph
    if graph_type == "dag":
        graph = pywhy_graphs.ADMG()
    elif graph_type == "admg":
        graph = pywhy_graphs.ADMG()
    elif graph_type == "cpdag":
        graph = pywhy_graphs.CPDAG()  # type: ignore
    elif graph_type == "pag":
        graph = pywhy_graphs.PAG()
    else:
        raise RuntimeError(
            f"The graph type {graph_type} is unrecognized. Please use one of "
            f"'dag', 'admg', 'cpdag', 'pag'."
        )

    graph.add_nodes_from(arr_idx)

    # convert each non-zero array entry combination into
    # an edge in the graph
    for udx in range(n_nodes):
        for vdx in range(n_nodes):
            if udx == vdx:
                continue

            endpoint_v = CLearnEndpoint(arr[vdx, udx])
            endpoint_u = CLearnEndpoint(arr[udx, vdx])
            u = arr_idx[udx]
            v = arr_idx[vdx]

            # First: check if there are two edges. If there
            if any(
                endpoint
                in [
                    CLearnEndpoint.ARROW_AND_ARROW,
                    CLearnEndpoint.TAIL_AND_ARROW,
                    CLearnEndpoint.TAIL_AND_TAIL,
                ]
                for endpoint in (endpoint_u, endpoint_v)
            ):
                # u -> v and u <-> v
                if (
                    endpoint_v == CLearnEndpoint.ARROW_AND_ARROW
                    and endpoint_u == CLearnEndpoint.TAIL_AND_ARROW
                ):
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                    graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                # u <- v and u <-> v
                elif (
                    endpoint_u == CLearnEndpoint.ARROW_AND_ARROW
                    and endpoint_v == CLearnEndpoint.TAIL_AND_ARROW
                ):
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                    graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                # u -> v and u -- v
                elif (endpoint_u == CLearnEndpoint.TAIL_AND_TAIL) and (
                    endpoint_v == CLearnEndpoint.TAIL_AND_ARROW
                ):
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
                # u <- v and u -- v
                elif (endpoint_v == CLearnEndpoint.TAIL_AND_TAIL) and (
                    endpoint_u == CLearnEndpoint.TAIL_AND_ARROW
                ):
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
                # u -- v and u <-> v
                elif (endpoint_v == CLearnEndpoint.TAIL_AND_ARROW) and (
                    endpoint_u == CLearnEndpoint.TAIL_AND_ARROW
                ):
                    graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
            # Else, there is only one edge between the two nodes and this is
            # either a DAG, or an equivalence class
            else:
                print("outside...", u, v, endpoint_u, endpoint_v)
                # there are no circle edges, implying this is not a PAG at least
                if all(endpoint != CLearnEndpoint.CIRCLE for endpoint in (endpoint_u, endpoint_v)):
                    # u <--> v
                    if (endpoint_v == CLearnEndpoint.ARROW) and (
                        endpoint_u == CLearnEndpoint.ARROW
                    ):
                        graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                    # u -> v
                    elif (endpoint_v == CLearnEndpoint.ARROW) and (
                        endpoint_u == CLearnEndpoint.TAIL
                    ):
                        graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                    # u <- v
                    elif (endpoint_u == CLearnEndpoint.ARROW) and (
                        endpoint_v == CLearnEndpoint.TAIL
                    ):
                        graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                    # u -- v
                    elif (endpoint_v == CLearnEndpoint.TAIL) and (
                        endpoint_u == CLearnEndpoint.TAIL
                    ):
                        graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
                else:
                    if not hasattr(graph, "circle_edge_name"):
                        raise RuntimeError(f"Graph {graph} is adding circular end points...")

                    # Endpoints contain a circle...
                    print(u, v, endpoint_u, endpoint_v)
                    if endpoint_u == CLearnEndpoint.CIRCLE and endpoint_v == CLearnEndpoint.TAIL:
                        print(endpoint_u, endpoint_v)
                        # u o- v
                        graph.add_edge(v, u, edge_type=graph.circle_edge_name)  # type: ignore
                    elif endpoint_u == CLearnEndpoint.TAIL and endpoint_v == CLearnEndpoint.CIRCLE:
                        print(endpoint_u, endpoint_v, "second")
                        # u -o v
                        graph.add_edge(u, v, edge_type=graph.circle_edge_name)  # type: ignore
                    elif endpoint_u == CLearnEndpoint.ARROW and endpoint_v == CLearnEndpoint.CIRCLE:
                        # u <-o v
                        graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                        graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                    elif endpoint_u == CLearnEndpoint.CIRCLE and endpoint_v == CLearnEndpoint.ARROW:
                        # u o-> v
                        graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                        graph.add_edge(v, u, edge_type=graph.circle_edge_name)
                    elif (
                        endpoint_u == CLearnEndpoint.CIRCLE and endpoint_v == CLearnEndpoint.CIRCLE
                    ):
                        # u o-o v
                        graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                        graph.add_edge(v, u, edge_type=graph.circle_edge_name)

    if graph_type == "dag":
        graph = graph.to_directed()
    return graph


def graph_to_arr(
    G,
    format: str = "causal-learn",
    node_order: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Node]]:
    """Convert a graph to a structured numpy array.

    Parameters
    ----------
    G : pywhy_nx.MixedEdgeGraph
        The mixed edge causal graph.
    format : str
        The format of the numpy array. One of 'causal-learn'. Default
        is 'causal-learn'.
    node_order : np.ndarray of shape (n_nodes,)
        The array of nodes in which we would like the order of the output array to
        be. See Notes for more information.

    Returns
    -------
    arr : np.ndarray of shape (n_nodes, n_nodes)
        The graph represented as a numpy array. See Notes for
        more information.
    arr_idx : List of length (n_nodes)
        The list of nodes representing the order of the nodes
        in the ``arr``.

    Notes
    -----
    The ``node_order`` parameter allows one to specify an array of nodes, with the order
    corresponding to how the rows/columns of ``arr`` and order of ``arr_idx`` should be.

    As of 09/05/2022, ``causal-learn`` does not explicitly support having a
    "tail and a tail" endpoint at one node. For example, an undirected edge and
    a directed edge: ``u -> v`` and ``u -- v``, indicating the presence of selection
    bias as well as a direct causal relationship between u and v. We add this possibility
    for encoding.

    Moreover, ``causal-learn`` does not support adding more than 2 edges between nodes.
    So simultaneous ``u -> v``, ``u -- v`` and ``u <-> v`` edges are not supported, although
    in principle they can be present in an ADMG.
    """
    # TODO: add option for exporting to pcalg
    if format == "causal-learn":
        arr, arr_idx = graph_to_clearn(G)

    if node_order is not None:
        new_order = np.searchsorted(node_order, arr_idx)
        arr = arr[np.ix_(new_order, new_order)]
        arr_idx = [arr_idx[idx] for idx in new_order]
    return arr, arr_idx
