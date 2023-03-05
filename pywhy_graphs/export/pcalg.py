from typing import Dict, List

import numpy as np

import pywhy_graphs
from pywhy_graphs.config import CLearnEndpoint, PCAlgCPDAGEndpoint, PCAlgPAGEndpoint
from pywhy_graphs.typing import Node

from .causallearn import graph_to_clearn


def pcalg_to_graph(arr, arr_idx: List[Node], amat_type: str):
    """Convert an array from R's pcalg into causal graph.

    Parameters
    ----------
    arr : array-like of shape (n_nodes, n_nodes)
        The array representing the causal graph with enumerations
        following that of `PCAlgPAGEndpoint`. Rows represent the starting
        node and the columns represent the ending node.
    arr_idx : List[Node] of length (n_nodes,)
        The names of the nodes that are assigned to the graph in order
        of the rows/columns of ``arr``.
    amat_type : str
        The type of graph in pcalg. One of ``{"pag", "cpdag"}``.

    Returns
    -------
    graph : causal graph
        The causal graph

    Notes
    -----
    See pcalg documentation for ``amatType``
    https://cran.r-project.org/web/packages/pcalg/pcalg.pdf. Copied here for convenience.

    Coding for type amat.cpdag:

    0: No edge or tail
    1: Arrowhead

    Note that the edgemark-code refers to the row index (as opposed adjacency matrices of
    type mag or pag). E.g.:

    amat[a,b] = 0  and  amat[b,a] = 1   implies a --> b.
    amat[a,b] = 1  and  amat[b,a] = 0   implies a <-- b.
    amat[a,b] = 0  and  amat[b,a] = 0   implies a     b.
    amat[a,b] = 1  and  amat[b,a] = 1   implies a --- b.

    Coding for type amat.pag:

    0: No edge
    1: Circle
    2: Arrowhead
    3: Tail

    Note that the edgemark-code refers to the column index (as opposed adjacency matrices of type
    dag or cpdag). E.g.:

    amat[a,b] = 2  and  amat[b,a] = 3   implies   a --> b.
    amat[a,b] = 3  and  amat[b,a] = 2   implies   a <-- b.
    amat[a,b] = 2  and  amat[b,a] = 2   implies   a <-> b.
    amat[a,b] = 1  and  amat[b,a] = 3   implies   a --o b.
    amat[a,b] = 0  and  amat[b,a] = 0   implies   a     b.
    """
    if amat_type not in ("pag", "cpdag"):
        raise RuntimeError(f'Only amat_types are "pag" and "cpdag", not {amat_type}')

    # instantiate the type of causal graph
    if amat_type == "cpdag":
        graph = pywhy_graphs.CPDAG()  # type: ignore
    elif amat_type == "pag":
        graph = pywhy_graphs.PAG()

    graph.add_nodes_from(arr_idx)

    # get all non-zero indices and add edge accordingly
    memo_map: Dict = dict()
    for idx, jdx in np.argwhere(arr != 0):
        arr_val = arr[idx, jdx]
        u, v = arr_idx[idx], arr_idx[jdx]

        # perform memoization to speed up loop
        if (idx, jdx) in memo_map:
            continue
        memo_map[(idx, jdx)] = None
        memo_map[(jdx, idx)] = None

        if amat_type == "pag":
            if arr_val == PCAlgPAGEndpoint.ARROW.value:
                # check other direction to determine if a bidirected edge
                if arr[jdx, idx] == PCAlgPAGEndpoint.ARROW.value:
                    graph.add_edge(u, v, edge_type=graph.bidirected_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.CIRCLE.value:
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
                    graph.add_edge(v, u, edge_type=graph.circle_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.TAIL.value:
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
            elif arr_val == PCAlgPAGEndpoint.TAIL.value:
                # check other direction to determine if a bidirected edge
                if arr[jdx, idx] == PCAlgPAGEndpoint.TAIL.value:
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.CIRCLE.value:
                    graph.add_edge(v, u, edge_type=graph.circle_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.ARROW.value:
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
            elif arr_val == PCAlgPAGEndpoint.CIRCLE.value:
                # check other direction to determine if a bidirected edge
                if arr[jdx, idx] == PCAlgPAGEndpoint.TAIL.value:
                    graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.CIRCLE.value:
                    graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                    graph.add_edge(v, u, edge_type=graph.circle_edge_name)
                elif arr[jdx, idx] == PCAlgPAGEndpoint.ARROW.value:
                    graph.add_edge(u, v, edge_type=graph.circle_edge_name)
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)
        elif amat_type == "cpdag":
            if arr_val == PCAlgCPDAGEndpoint.ARROW.value:
                # check other direction to determine if a bidirected edge
                if arr[jdx, idx] == PCAlgCPDAGEndpoint.ARROW.value:
                    graph.add_edge(u, v, edge_type=graph.undirected_edge_name)
                else:
                    graph.add_edge(u, v, edge_type=graph.directed_edge_name)
            elif arr_val == PCAlgCPDAGEndpoint.NULL.value:
                # check other direction to determine if a bidirected edge
                if arr[jdx, idx] == PCAlgCPDAGEndpoint.ARROW.value:
                    graph.add_edge(v, u, edge_type=graph.directed_edge_name)

    return graph


def graph_to_pcalg(causal_graph):
    """Convert causal graph to a pcalg type adjacency array.

    Parameters
    ----------
    causal_graph : instance of causal graph
        The causal graph that is represented in pywhy. Either CPDAG, or PAG.

    Returns
    -------
    numpy_graph : array-like of shape (n_nodes, n_nodes)
        The numpy array that represents the graph. The values representing edges
        are mapped according to a pre-defined set of values. See Notes.

    See Also
    --------
    pcalg_to_graph

    Notes
    -----
    See :func:`pcalg_to_graph` for information on how edges are represented in pcalg.
    """
    if not isinstance(causal_graph, (pywhy_graphs.CPDAG, pywhy_graphs.PAG)):
        raise RuntimeError(f"Causal graph must be one of CPDAG or PAG, not {causal_graph}.")
    elif isinstance(causal_graph, pywhy_graphs.CPDAG):
        amat_type = "cpdag"
    elif isinstance(causal_graph, pywhy_graphs.PAG):
        amat_type = "pag"

    # first convert to causal-learn array
    clearn_arr, _ = graph_to_clearn(causal_graph)

    # the pcalg array is in a different format
    clearn_arr = clearn_arr.T

    # now map all values to their respective pcalg values
    seen_idx = dict()
    for (idx, jdx) in np.argwhere(clearn_arr != 0):
        if (idx, jdx) in seen_idx or (jdx, idx) in seen_idx:
            continue

        seen_idx[(idx, jdx)] = None
        if amat_type == "cpdag":
            if (
                clearn_arr[idx, jdx] == CLearnEndpoint.TAIL.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.TAIL.value
            ):
                # --
                clearn_arr[idx, jdx] = PCAlgCPDAGEndpoint.ARROW.value
                clearn_arr[jdx, idx] = PCAlgCPDAGEndpoint.ARROW.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.ARROW.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.TAIL.value
            ):
                # ->
                clearn_arr[idx, jdx] = PCAlgCPDAGEndpoint.ARROW.value
                clearn_arr[jdx, idx] = PCAlgCPDAGEndpoint.NULL.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.TAIL.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.ARROW.value
            ):
                # <-
                clearn_arr[idx, jdx] = PCAlgCPDAGEndpoint.NULL.value
                clearn_arr[jdx, idx] = PCAlgCPDAGEndpoint.ARROW.value
        if amat_type == "pag":
            if (
                clearn_arr[idx, jdx] == CLearnEndpoint.ARROW.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.NULL.value
            ):
                # ->
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.ARROW.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.NULL.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.NULL.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.ARROW.value
            ):
                # <-
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.NULL.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.ARROW.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.ARROW.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.ARROW.value
            ):
                # <->
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.ARROW.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.ARROW.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.ARROW.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.CIRCLE.value
            ):
                # o->
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.ARROW.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.CIRCLE.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.CIRCLE.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.ARROW.value
            ):
                # <-o
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.CIRCLE.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.ARROW.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.TAIL.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.TAIL.value
            ):
                # --
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.TAIL.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.TAIL.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.CIRCLE.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.CIRCLE.value
            ):
                # o-o
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.CIRCLE.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.CIRCLE.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.CIRCLE.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.TAIL.value
            ):
                # o--
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.CIRCLE.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.TAIL.value
            elif (
                clearn_arr[idx, jdx] == CLearnEndpoint.TAIL.value
                and clearn_arr[jdx, idx] == CLearnEndpoint.CIRCLE.value
            ):
                # --o
                clearn_arr[idx, jdx] = PCAlgPAGEndpoint.TAIL.value
                clearn_arr[jdx, idx] = PCAlgPAGEndpoint.CIRCLE.value
    return clearn_arr
