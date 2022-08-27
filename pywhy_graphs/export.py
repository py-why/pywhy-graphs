from copy import deepcopy

import networkx as nx
import numpy as np

from pywhy_graphs import ADMG

EDGE_TO_VALUE_MAPPING = {
    None: 0,
    "directed": 1,
    "undirected": 2,
    "bidirected": 3,
    "circle": 4,
}


def to_digraph(graph: nx.MixedEdgeGraph):
    """Convert causal graph to a uni-edge networkx directed graph.

    Parameters
    ----------
    graph : MixedEdgeGraph
        A causal mixed-edge graph.

    Returns
    -------
    G : nx.DiGraph | nx.MultiDiGraph
        The networkx directed graph with multiple edges with edge
        attributes indicating via the keyword "type", which type of
        causal edge it is.
    """
    if len(graph.get_graphs()) == 1:
        G = nx.DiGraph()
    else:
        G = nx.MultiDiGraph()

    # preserve the name
    G.graph.update(deepcopy(graph.graph))
    graph_type = type(graph).__name__  # GRAPH_TYPE[type(causal_graph)]
    G.graph["graph_type"] = graph_type

    G.add_nodes_from((n, deepcopy(d)) for n, d in graph.nodes.items())

    # add all the edges
    for edge_type, edge_adj in graph.adj.items():
        # replace edge marks with their appropriate string representation
        attr = {"type": edge_type}
        G.add_edges_from(
            (u, v, deepcopy(d), attr.items())
            for u, nbrs in edge_adj.items()
            for v, d in nbrs.items()
        )
    return G


def to_numpy(causal_graph):
    """Convert causal graph to a numpy adjacency array.

    Parameters
    ----------
    causal_graph : instance of DAG
        The causal graph.

    Returns
    -------
    numpy_graph : np.ndarray of shape (n_nodes, n_nodes)
        The numpy array that represents the graph. The values representing edges
        are mapped according to a pre-defined set of values. See Notes.

    Notes
    -----
    The adjacency matrix is defined where the ijth entry of ``numpy_graph`` has a
    non-zero entry if there is an edge from i to j. The ijth entry is symmetric with the
    jith entry if the edge is 'undirected', or 'bidirected'. Then specific edges are
    mapped to the following values:

        - directed edge (->): 1
        - undirected edge (--): 2
        - bidirected edge (<->): 3
        - circle endpoint (-o): 4

    Circle endpoints can be symmetric, but they can also contain a tail, or a directed
    edge at the other end.
    """
    if isinstance(causal_graph, ADMG):
        raise RuntimeError("Converting ADMG to numpy format is not supported.")

    # master list of nodes is in the internal dag
    node_list = causal_graph.nodes
    n_nodes = len(node_list)

    numpy_graph = np.zeros((n_nodes, n_nodes))
    bidirected_graph_arr = None
    graph_map = dict()
    for edge_type, graph in causal_graph.get_graphs():
        # handle bidirected edge separately
        if edge_type == "bidirected":
            bidirected_graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
            continue

        # convert internal graph to a numpy array
        graph_arr = nx.to_numpy_array(graph, nodelist=node_list)
        graph_map[edge_type] = graph_arr

    # ADMGs can have two edges between any 2 nodes
    if type(causal_graph).__name__ == "ADMG":
        # we handle this case separately from the other graphs
        if len(graph_map) != 1:
            raise AssertionError("The number of graph maps should be 1...")

        # set all bidirected edges with value 10
        bidirected_graph_arr[bidirected_graph_arr != 0] = 10
        numpy_graph += bidirected_graph_arr
        numpy_graph += graph_arr
    else:
        # map each edge to an edge value
        for name, graph_arr in graph_map.items():
            graph_arr[graph_arr != 0] = EDGE_TO_VALUE_MAPPING[name]
            numpy_graph += graph_arr

        # bidirected case is handled separately
        if bidirected_graph_arr is not None:
            numpy_graph += bidirected_graph_arr

    return numpy_graph
