import networkx as nx

import pywhy_graphs.networkx as pywhy_nx


def test_m_separation():

    # 0 -> 1 -> 2 -> 3 -> 4; 2 -> 4; 2 <-> 3
    digraph = nx.path_graph(4, create_using=nx.DiGraph)
    digraph.add_edge(2, 4)
    bigraph = nx.Graph([(2, 3)])
    bigraph.add_nodes_from(digraph)
    G = pywhy_nx.MixedEdgeGraph([digraph, bigraph], ["directed", "bidirected"])

    nx_G = pywhy_nx.bidirected_to_unobserved_confounder(G)

    assert isinstance(nx_G, nx.DiGraph)

    expected_G = digraph.copy()
    expected_G.add_edge("U0", 2)
    expected_G.add_edge("U0", 3)
    assert nx.is_isomorphic(expected_G, nx_G)
