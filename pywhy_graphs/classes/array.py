from typing import Callable, List, Tuple

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

import pywhy_graphs
from pywhy_graphs.config import CLearnEndpoint, EdgeType
from pywhy_graphs.typing import Node


def has_multiple_edges(G: pywhy_graphs.ADMG, u: Node, v: Node) -> bool:
    edge_count = 0
    for _, graph in G.get_graphs().items():
        if graph.has_edge(u, v):
            edge_count += 1

    return edge_count > 1


def edge_types(G: pywhy_graphs.ADMG, u: Node, v: Node) -> List[str]:
    edge_types = []
    for edge_type, graph in G.get_graphs().items():
        if graph.has_edge(u, v):
            edge_types.append(edge_type)
    return edge_types


def _graph_to_clearn_arr(G: nx.MixedEdgeGraph) -> Tuple[ArrayLike, List[Node]]:
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
                if edge_type == EdgeType.DIRECTED:
                    # ->
                    endpoint_v = CLearnEndpoint.ARROW
                    endpoint_u = CLearnEndpoint.TAIL
                elif edge_type == EdgeType.BIDIRECTED:
                    # <->
                    endpoint_v = CLearnEndpoint.ARROW
                    endpoint_u = CLearnEndpoint.ARROW
                elif edge_type == EdgeType.UNDIRECTED:
                    # --
                    endpoint_v = CLearnEndpoint.TAIL
                    endpoint_u = CLearnEndpoint.TAIL
            elif len(uv_edge_types) == 2:
                if (EdgeType.DIRECTED in uv_edge_types) and (EdgeType.BIDIRECTED in uv_edge_types):
                    # u -> v and u <-> v
                    endpoint_v = CLearnEndpoint.ARROW_AND_ARROW
                    endpoint_u = CLearnEndpoint.TAIL_AND_ARROW
                elif (EdgeType.DIRECTED in uv_edge_types) and (
                    EdgeType.UNDIRECTED in uv_edge_types
                ):
                    # u -> v and u -- v
                    endpoint_v = CLearnEndpoint.TAIL_AND_ARROW
                    endpoint_u = CLearnEndpoint.TAIL_AND_TAIL
                elif (EdgeType.BIDIRECTED in uv_edge_types) and (
                    EdgeType.UNDIRECTED in uv_edge_types
                ):
                    # u -- v and u <-> v
                    endpoint_v = CLearnEndpoint.TAIL_AND_ARROW
                    endpoint_u = CLearnEndpoint.TAIL_AND_ARROW
            else:
                raise RuntimeError(
                    f"Causal-learn does not support more than two types of edges between nodes. There are "
                    f"{len(uv_edge_types)} edge types between {u} and {v}."
                )

            # set the array to the endpoint values
            arr[udx, vdx] = endpoint_v.value
            arr[vdx, udx] = endpoint_u.value

    return arr, arr_idx


def _infer_graph_from_causallearn(arr: ArrayLike) -> Callable:
    unique_edge_nums = np.unique(arr)
    n_nodes = arr.shape[0]

    # check if there are two edges
    if any(
        CLearnEndpoint(endpoint)
        in [
            CLearnEndpoint.ARROW_AND_ARROW,
            CLearnEndpoint.TAIL_AND_ARROW,
            CLearnEndpoint.TAIL_AND_TAIL,
        ]
        for endpoint in unique_edge_nums
    ):
        graph_func = pywhy_graphs.ADMG
    elif any(CLearnEndpoint(endpoint) == CLearnEndpoint.TAIL):
        pass
    # convert each non-zero array entry combination into
    # an edge in the graph
    for udx, vdx in np.triu_indices(n_nodes, k=1):
        endpoint_u = CLearnEndpoint(arr[vdx, udx])
        endpoint_v = CLearnEndpoint(arr[udx, vdx])


def _graph_to_pcalg_arr(G: pywhy_graphs.ADMG) -> Tuple[ArrayLike, List[Node]]:
    pass


def clearn_arr_to_graph(arr: ArrayLike, arr_idx: List[Node], graph_type: str) -> nx.MixedEdgeGraph:
    """Convert causal-learn array to a graph object.

    Parameters
    ----------
    arr : ArrayLike of shape (n_nodes, n_nodes)
        The causal-learn array encoding the endpoints between nodes.
    arr_idx : List[Node] of length (n_nodes)
        The array index, which stores the name of the n_nodes in order of their
        rows/columns in ``arr``.
    graph_type : str, optional
        The type of causal graph. Must be one of 'dag', 'admg', 'cpdag', 'pag'.

    Returns
    -------
    graph : nx.MixedEdgeGraph
        The causal graph.
    """
    if arr.shape[0] != arr.shape[1]:
        raise RuntimeError("Only square arrays are convertable to pywhy-graphs.")

    n_nodes = arr.shape[0]
    if len(arr_idx) != n_nodes:
        raise RuntimeError(
            f"The number of node names in order of the array rows/columns, {len(arr_idx)} "
            f"should match the number of rows/columns in array, {n_nodes}."
        )

    unique_edge_nums = np.unique(arr)
    if any(CLearnEndpoint(num) not in CLearnEndpoint for num in unique_edge_nums):
        raise RuntimeError(f"Some entries of array are not causal-learn specified.")

    # TODO: enable us to infer the type?
    if graph_type == "dag":
        graph_func = nx.DiGraph
    elif graph_type == "admg":
        graph_func = pywhy_graphs.ADMG
    elif graph_type == "cpdag":
        graph_func = pywhy_graphs.CPDAG
    elif graph_type == "pag":
        graph_func = pywhy_graphs.PAG
    else:
        raise RuntimeError(
            f"The graph type {graph_type} is unrecognized. Please use one of "
            f"'dag', 'admg', 'cpdag', 'pag'."
        )
    # instantiate the graph
    graph = graph_func()

    # convert each non-zero array entry combination into
    # an edge in the graph
    for udx, vdx in np.triu_indices(n_nodes, k=1):
        endpoint_u = CLearnEndpoint(arr[vdx, udx])
        endpoint_v = CLearnEndpoint(arr[udx, vdx])
        u = arr_idx[udx]
        v = arr_idx[vdx]

        # check if there are two edges
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
        # there is only one edge between the two nodes
        else:
            if not any(endpoint in CLearnEndpoint.CIRCLE for endpoint in (endpoint_u, endpoint_v)):
                # u <--> v
                if (endpoint_v == CLearnEndpoint.ARROW) and (endpoint_u == CLearnEndpoint.ARROW):
                    graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                # u -> v
                elif (endpoint_v == CLearnEndpoint.ARROW) and (endpoint_u == CLearnEndpoint.TAIL):
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                # u <- v
                elif (endpoint_u == CLearnEndpoint.ARROW) and (endpoint_v == CLearnEndpoint.TAIL):
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                # u -- v
                elif (endpoint_v == CLearnEndpoint.TAIL) and (endpoint_u == CLearnEndpoint.TAIL):
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
            else:
                # Endpoints contain a circle...
                # u o- v
                if endpoint_u == CLearnEndpoint.CIRCLE:
                    graph.add_edge(v, u, edge_type=graph.circle_edge_name)
                elif endpoint_u == CLearnEndpoint.ARROW:
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
                elif endpoint_u == CLearnEndpoint.TAIL:
                    graph.add_edge(v, u, edge_type=graph.undirected_edge_name)

                # u -o v
                if endpoint_v == CLearnEndpoint.CIRCLE:
                    graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                elif endpoint_v == CLearnEndpoint.ARROW:
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                elif endpoint_v == CLearnEndpoint.TAIL:
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
    return graph


def graph_to_arr(
    G: nx.MixedEdgeGraph,
    format: str='causal-learn'
) -> Tuple[ArrayLike, List[Node]]:
    """Convert a graph to a structured numpy array.

    Parameters
    ----------
    G : nx.MixedEdgeGraph
        _description_

    Returns
    -------
    arr : ArrayLike of shape (n_nodes, n_nodes)
        The graph represented as a numpy array. See Notes for
        more information.
    arr_idx : List of length (n_nodes)
        The list of nodes representing the order of the nodes
        in the ``arr``.

    Notes
    -----
    ``pcalg`` does not encode
    """
    if format == 'causal-learn':
        arr, arr_idx = _graph_to_clearn_arr(G)
    elif format == 'pcalg':
        arr, arr_idx = _graph_to_pcalg_arr(G)
    return arr, arr_idx
