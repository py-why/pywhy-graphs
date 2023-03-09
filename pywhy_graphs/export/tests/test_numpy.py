import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal

from pywhy_graphs import ADMG, PAG
from pywhy_graphs.export import graph_to_numpy, numpy_to_graph

# setup graphs


def test_to_numpy_admg():
    """Test conversion of ADMG to numpy array and back."""
    G = ADMG()
    G.add_nodes_from(["x", "y", "z"])
    G.add_edge("x", "y", G.directed_edge_name)

    # the roundtrip should be correct
    G_arr = graph_to_numpy(G)
    G_copy = numpy_to_graph(G_arr, list(G.nodes), ADMG)
    for edge_type, graph in G.get_graphs().items():
        assert nx.is_isomorphic(graph, G_copy.get_graphs(edge_type=edge_type))

    G.add_edge("x", "y", G.bidirected_edge_name)
    # the roundtrip should be correct
    G_arr = graph_to_numpy(G)
    G_copy = numpy_to_graph(G_arr, list(G.nodes), ADMG)
    expected_arr = np.array([[0, 21, 0], [20, 0, 0], [0, 0, 0]])
    assert_array_equal(G_arr, expected_arr)

    for edge_type, graph in G.get_graphs().items():
        assert nx.is_isomorphic(graph, G_copy.get_graphs(edge_type=edge_type))


def test_to_numpy_pag():
    G = PAG()
    G.add_nodes_from(["x", "y", "z"])
    G.add_edge("x", "y", G.directed_edge_name)
    G.add_edge("x", "z", G.circle_edge_name)

    G_arr = graph_to_numpy(G)
    G_copy = numpy_to_graph(G_arr, list(G.nodes), "pag")
    for edge_type, graph in G.get_graphs().items():
        assert nx.is_isomorphic(graph, G_copy.get_graphs(edge_type=edge_type))
