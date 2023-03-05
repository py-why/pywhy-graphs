from typing import List

import networkx as nx
import numpy as np

import pywhy_graphs
from pywhy_graphs.config import EDGE_TO_VALUE_MAPPING, VALUE_TO_EDGE_MAPPING
from pywhy_graphs.typing import Node


def numpy_to_graph(arr, arr_idx: List[Node], graph_type):
    """Convert an enumerated numpy array into causal graph.

    Parameters
    ----------
    arr : array-like of shape (n_nodes, n_nodes)
        The array representing the causal graph with enumerations
        following that of `EDGE_TO_VALUE_MAPPING`.
    arr_idx : List[Node] of length (n_nodes,)
        The names of the nodes that are assigned to the graph in order
        of the rows/columns of ``arr``.
    graph_type : str
        The type of causal graph to construct. One of ('pag', 'cpdag',
        'admg', 'dag').

    Returns
    -------
    graph : causal graph
        The causal graph
    """
    # instantiate the type of causal graph
    if graph_type == "dag":
        graph = pywhy_graphs.ADMG()
    elif graph_type == "admg":
        graph = pywhy_graphs.ADMG()
    elif graph_type == "cpdag":
        graph = pywhy_graphs.CPDAG()  # type: ignore
    elif graph_type == "pag":
        graph = pywhy_graphs.PAG()
    elif graph_type in [pywhy_graphs.ADMG, pywhy_graphs.CPDAG, pywhy_graphs.PAG]:
        graph = graph_type()
    else:
        raise RuntimeError(
            f"The graph type {graph_type} is unrecognized. Please use one of "
            f"'dag', 'admg', 'cpdag', 'pag'."
        )

    graph.add_nodes_from(arr_idx)

    # get all non-zero indices and add edge accordingly
    for idx, jdx in np.argwhere(arr != 0):
        arr_val = arr[idx, jdx]
        u, v = arr_idx[idx], arr_idx[jdx]

        if arr_val >= EDGE_TO_VALUE_MAPPING["bidirected"]:
            # we have a bidirected edge at least
            edge_type = VALUE_TO_EDGE_MAPPING[EDGE_TO_VALUE_MAPPING["bidirected"]]
            graph.add_edge(u, v, edge_type=edge_type)
            if arr_val % EDGE_TO_VALUE_MAPPING["bidirected"] > 0:
                arr_val -= EDGE_TO_VALUE_MAPPING["bidirected"]
                edge_type = VALUE_TO_EDGE_MAPPING[arr_val]
                graph.add_edge(u, v, edge_type=edge_type)
        elif arr_val >= EDGE_TO_VALUE_MAPPING["undirected"]:
            # we have an undirected edge at least
            edge_type = VALUE_TO_EDGE_MAPPING[EDGE_TO_VALUE_MAPPING["undirected"]]
            graph.add_edge(u, v, edge_type=edge_type)
            if np.mod(arr_val, EDGE_TO_VALUE_MAPPING["undirected"]) > 0:
                arr_val -= EDGE_TO_VALUE_MAPPING["undirected"]
                VALUE_TO_EDGE_MAPPING[arr_val]
                graph.add_edge(u, v, edge_type=edge_type)
        else:
            # we only have a single edge
            edge_type = VALUE_TO_EDGE_MAPPING[arr_val]
            graph.add_edge(u, v, edge_type=edge_type)

    if graph_type == "dag":
        graph = graph.to_directed()
    return graph


def graph_to_numpy(causal_graph):
    """Convert causal graph to a numpy adjacency array.

    Parameters
    ----------
    causal_graph : instance of causal graph
        The causal graph that is represented in pywhy.

    Returns
    -------
    numpy_graph : array-like of shape (n_nodes, n_nodes)
        The numpy array that represents the graph. The values representing edges
        are mapped according to a pre-defined set of values. See Notes.

    Examples
    --------
    > arr = np.array([
        [0, 21, 0],
        [20, 0, 0],
        [0, 0, 0]
    ])
    > nodelist = ['x', 'y', 'z']
    > bow_graph = numpy_to_graph(arr, nodelist, 'admg')
    > print(bow_graph.edges())

    Notes
    -----
    The adjacency matrix is defined where the ijth entry of ``numpy_graph`` has a
    non-zero entry if there is an edge from i to j. The ijth entry is symmetric with the
    jith entry if the edge is 'undirected', or 'bidirected'. Then specific edges are
    mapped to the following values:

        - directed edge (->): 1
        - circle endpoint (-o): 2
        - undirected edge (--): 10
        - bidirected edge (<->): 20

    Circle endpoints can be symmetric, but they can also contain a tail, or a directed
    edge at the other end. See `EDGE_TO_VALUE_MAPPING`. This corresponds to the
    output of the `pcalg <>`_ package.

    **How are multiple edges between same pair of nodes handled?**

    In ADMGs, multiple edges between the same pairs of nodes are allowed. Since we
    map edges between pairs of nodes to numerical values, we have to treat
    undirected and bidirected edges separately, since one can have a directed edge
    and either an undirected, or bidirected edge present. Therefore for example,
    if there is a directed edge :math:`X \\rightarrow Y` and also a bidirected
    edge :math:`X \\leftrightarrow Y`, then the numpy array element corresponding
    to (X, Y) would have the value 21, indicating uniquely a directed edge and a
    bidirected edge. Note, this is not an issue for any other common causal graph
    class because there only one edge is supported between any two nodes.
    """
    undirected_edge_name = "undirected"
    bidirected_edge_name = "bidirected"

    # master list of nodes is in the internal dag
    node_list = causal_graph.nodes
    n_nodes = len(node_list)

    numpy_graph = np.zeros((n_nodes, n_nodes))
    bidirected_graph_arr = None
    undirected_graph_arr = None

    graph_map = dict()
    for edge_type, graph in causal_graph.get_graphs().items():
        # handle "undirected" type graphs separately
        # handle bidirected edge separately
        if edge_type == bidirected_edge_name:
            bidirected_graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
            continue
        if edge_type == undirected_edge_name:
            undirected_graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
            continue

        # convert internal graph to a numpy array
        graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
        graph_arr[graph_arr != 0] = EDGE_TO_VALUE_MAPPING[edge_type]
        graph_map[edge_type] = graph_arr

    # ADMGs can have two edges between any 2 nodes
    if type(causal_graph).__name__ == "ADMG":
        # we handle this case separately from the other graphs
        if len(graph_map) != 1:
            raise AssertionError(f"The number of graph maps should be 1, not {len(graph_map)}...")

        # set all bidirected edges with value 10
        bidirected_graph_arr[bidirected_graph_arr != 0] = EDGE_TO_VALUE_MAPPING[
            bidirected_edge_name
        ]
        undirected_graph_arr[undirected_graph_arr != 0] = EDGE_TO_VALUE_MAPPING[
            undirected_edge_name
        ]
        numpy_graph += bidirected_graph_arr
        numpy_graph += graph_arr
    else:
        # map each edge to an edge value
        for _, graph_arr in graph_map.items():
            numpy_graph += graph_arr
    return numpy_graph
