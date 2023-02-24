from copy import deepcopy

import networkx as nx

import pywhy_graphs.networkx as pywhy_nx


def graph_to_digraph(graph: pywhy_nx.MixedEdgeGraph):
    """Convert causal graph to a uni-edge networkx directed graph.

    Parameters
    ----------
    graph : pywhy_nx.MixedEdgeGraph
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
