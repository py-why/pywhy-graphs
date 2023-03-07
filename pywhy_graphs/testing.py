import networkx as nx


def assert_mixed_edge_graphs_isomorphic(G1, G2):
    """Test that two mixed-edge graphs are isomorphic.

    Mixed-edge graphs are isomorphic if for each edge type,
    their edge-type subgraphs are isomorphic.

    Parameters
    ----------
    G1 : causal graph
        A graph with mixed edges.
    G2 : causal graph
        A graph with mixed edges.

    Returns
    -------
    bool : Whether or not the two graphs are isomorphic.
    """
    for edge_type, graph in G1.get_graphs().items():
        if edge_type not in G2.edge_types:
            return False

        if not nx.is_isomorphic(graph, G2.get_graphs(edge_type)):
            return False
    return True
