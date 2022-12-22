import networkx as nx

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx


def test_acyclification():
    """Test acyclification procedure as specified in :footcite:`Mooij2020cyclic`.

    Tests the graphs as presented in Figure 2.
    """
    # cycle with x2 -> x4 -> x6 -> x5 -> x3 -> x4
    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x4", "x6"),  # start of cycle
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),  # end of cycle
            ("x6", "x7"),
        ]
    )
    bidirected_edges = nx.Graph([("x1", "x3")])
    G = pywhy_nx.MixedEdgeGraph([directed_edges, bidirected_edges], ["directed", "bidirected"])
    acyclic_G = pywhy_graphs.acyclification(G)

    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x6", "x7"),
            ("x2", "x3"),
            ("x2", "x5"),
            ("x2", "x4"),
            ("x2", "x6"),
        ]
    )
    bidirected_edges = nx.Graph(
        [
            ("x1", "x3"),
            ("x4", "x6"),
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),
            ("x4", "x5"),
            ("x3", "x6"),
            ("x1", "x3"),
            ("x1", "x5"),
            ("x1", "x4"),
            ("x1", "x6"),
        ]
    )
    expected_G = pywhy_nx.MixedEdgeGraph(
        [directed_edges, bidirected_edges], ["directed", "bidirected"]
    )

    for edge_type, graph in acyclic_G.get_graphs().items():
        expected_graph = expected_G.get_graphs(edge_type)
        assert nx.is_isomorphic(graph, expected_graph)
