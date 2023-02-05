import networkx as nx
import pytest
from networkx.exception import NetworkXError

import pywhy_graphs.networkx as pywhy_nx


def test_m_separation():
    digraph = nx.path_graph(4, create_using=nx.DiGraph)
    digraph.add_edge(2, 4)
    bigraph = nx.Graph([(2, 3)])
    bigraph.add_nodes_from(digraph)
    G = pywhy_nx.MixedEdgeGraph([digraph, bigraph], ["directed", "bidirected"])

    # error should be raised if someone does not use a MixedEdgeGraph
    with pytest.raises(NetworkXError, match="m-separation should only be run on a MixedEdgeGraph"):
        pywhy_nx.m_separated(digraph, {0}, {1}, set())

    # error should be raised if the directed edges form a cycle
    G_error = G.copy()
    G_error.add_edge(4, 2, "directed")
    with pytest.raises(NetworkXError, match="directed edge graph should be directed acyclic"):
        pywhy_nx.m_separated(G_error, {0}, {3}, set())

    # if passing in non-default names for edge types, then m_separated will not work
    G_error = G.copy()
    G_error._edge_graphs["bi-directed"] = G_error.get_graphs("bidirected")
    G_error._edge_graphs.pop("bidirected")
    with pytest.raises(
        NetworkXError,
        match="m-separation only works on graphs with directed, bidirected, and undirected edges.",
    ):
        pywhy_nx.m_separated(G_error, {0}, {3}, set())
    assert not pywhy_nx.m_separated(G_error, {0}, {3}, set(), bidirected_edge_name="bi-directed")

    # basic d-separation statements based on blocking paths should work
    assert not pywhy_nx.m_separated(G, {0}, {3}, set())
    assert pywhy_nx.m_separated(G, {0}, {3}, {1})

    # conditioning on a collider via bidirected edge opens the collider
    assert not pywhy_nx.m_separated(G, {0}, {3}, {2})
    assert pywhy_nx.m_separated(G, {0}, {3}, {1, 2})

    # conditioning on a descendant of a collider via bidirected edge opens the collider
    assert not pywhy_nx.m_separated(G, {0}, {3}, {4})

    # check that works when there are only bidirected edges present
    bigraph = nx.Graph([(1, 2), (2, 3)])
    G = pywhy_nx.MixedEdgeGraph([bigraph], ["bidirected"])

    assert pywhy_nx.m_separated(G, {1}, {3}, set())
    assert not pywhy_nx.m_separated(G, {1}, {3}, {2})

    # check that 1 _|_ 5 in graph 1 - 2 -> 3 <- 4 <-> 5
    digraph = nx.DiGraph()
    digraph.add_nodes_from([1, 2, 3, 4, 5])
    digraph.add_edge(2, 3)
    digraph.add_edge(4, 3)
    bigraph = nx.Graph([(4, 5)])
    bigraph.add_nodes_from(digraph)
    ungraph = nx.Graph([(1, 2)])
    ungraph.add_nodes_from(digraph)
    G = pywhy_nx.MixedEdgeGraph(
        [digraph, bigraph, ungraph], ["directed", "bidirected", "undirected"]
    )

    assert pywhy_nx.m_separated(G, {1}, {4}, set())

    # check that 1 not _|_ 3 in graph 1 - 2 - 3
    ungraph = nx.Graph([(1, 2), (2, 3)])
    G = pywhy_nx.MixedEdgeGraph([ungraph], ["undirected"])

    assert not pywhy_nx.m_separated(G, {1}, {3}, set())

    assert pywhy_nx.m_separated(G, {1}, {3}, {2})

    G.add_edge(1, 3, "undirected")

    assert not pywhy_nx.m_separated(G, {1}, {3}, {2})

    # check that in a graph with no edges, everything is m-separated
    digraph = nx.DiGraph()
    digraph.add_nodes_from([1, 2, 3])
    bigraph = nx.Graph()
    bigraph.add_nodes_from(digraph)
    ungraph = nx.Graph()
    ungraph.add_nodes_from(ungraph)
    G = pywhy_nx.MixedEdgeGraph(
        [digraph, bigraph, ungraph], ["directed", "bidirected", "undirected"]
    )

    assert pywhy_nx.m_separated(G, {1}, {3}, {2})
    assert pywhy_nx.m_separated(G, {1}, {2}, set())
