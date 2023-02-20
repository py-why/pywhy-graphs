import logging

import networkx as nx
import pytest
from networkx.exception import NetworkXError

import pywhy_graphs.networkx as pywhy_nx


@pytest.fixture
def fig5_vanderzander():

    nodes = ["V_1", "X", "V_2", "Y", "Z_1", "Z_2"]

    digraph = nx.DiGraph()
    digraph.add_nodes_from(nodes)
    ungraph = nx.Graph()
    ungraph.add_nodes_from(nodes)
    bigraph = nx.Graph()
    bigraph.add_nodes_from(nodes)

    digraph.add_edge("V_1", "X")
    digraph.add_edge("Z_1", "X")
    digraph.add_edge("X", "V_2")
    digraph.add_edge("Y", "V_2")
    digraph.add_edge("Z_2", "Y")
    digraph.add_edge("Z_2", "Z_1")

    G = pywhy_nx.MixedEdgeGraph(
        [digraph, ungraph, bigraph], ["directed", "undirected", "bidirected"]
    )

    return G


@pytest.fixture
def modified_fig5_vanderzander():

    nodes = ["V_1", "X", "V_2", "Y", "Z_1", "Z_2"]

    digraph = nx.DiGraph()
    digraph.add_nodes_from(nodes)
    ungraph = nx.Graph()
    ungraph.add_nodes_from(nodes)
    bigraph = nx.Graph()
    bigraph.add_nodes_from(nodes)

    digraph.add_edge("V_1", "X")
    digraph.add_edge("Z_1", "X")
    digraph.add_edge("X", "V_2")
    digraph.add_edge("Y", "V_2")
    digraph.add_edge("Z_2", "Y")
    digraph.add_edge("Z_2", "Z_1")

    G = pywhy_nx.MixedEdgeGraph(
        [digraph, ungraph, bigraph], ["directed", "undirected", "bidirected"]
    )

    return G


def test_m_separation():
    logging.getLogger().setLevel(logging.DEBUG)

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

    # check that m-sep in graph with all kinds of edges
    # e.g. 1 _|_ 5 in graph 1 - 2 -> 3 <- 4 <-> 5
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

    # e.g. 1 _|_ 5 | 7 in 1 - 2 -> 3 <-> 4 - 5, 3 -> 6, 2 - 7 <-> 5
    digraph = nx.DiGraph()
    digraph.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    digraph.add_edges_from([(2, 3), (3, 6)])
    bigraph = nx.Graph([(3, 4), (7, 5)])
    bigraph.add_nodes_from(digraph)
    ungraph = nx.Graph([(1, 2), (4, 5), (2, 7)])
    ungraph.add_nodes_from(digraph)
    G = pywhy_nx.MixedEdgeGraph(
        [digraph, bigraph, ungraph], ["directed", "bidirected", "undirected"]
    )

    assert pywhy_nx.m_separated(G, {1}, {5}, {7})
    assert not pywhy_nx.m_separated(G, {1}, {5}, set())
    assert not pywhy_nx.m_separated(G, {1}, {5}, {6})
    assert not pywhy_nx.m_separated(G, {1}, {5}, {6, 7})

    # check m-sep works in undirected graphs:
    # e.g. that 1 not _|_ 3 in graph 1 - 2 - 3
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

    # check fig 6 of Zhang 2008

    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C", "D"])
    digraph.add_edge("A", "C")
    digraph.add_edge("C", "D")
    digraph.add_edge("B", "D")
    bigraph = nx.Graph()
    bigraph.add_edge("A", "B")
    G = pywhy_nx.MixedEdgeGraph([digraph, bigraph], ["directed", "bidirected"])
    assert not pywhy_nx.m_separated(G, {"A"}, {"D"}, {"C"})
    assert pywhy_nx.m_separated(G, {"A"}, {"D"}, {"B", "C"})

    assert pywhy_nx.m_separated(G, {"B"}, {"C"}, {"A"})
    assert not pywhy_nx.m_separated(G, {"B"}, {"C"}, {"A", "D"})
    assert not pywhy_nx.m_separated(G, {"B"}, {"C"}, set())

    # check more complicated ADMGs

    # check inducing paths behave correctly
    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C", "D"])
    digraph.add_edge("B", "C")
    digraph.add_edge("C", "D")
    bigraph = nx.Graph()
    bigraph.add_edge("A", "B")
    bigraph.add_edge("B", "C")
    G = pywhy_nx.MixedEdgeGraph([digraph, bigraph], ["directed", "bidirected"])

    assert not pywhy_nx.m_separated(G, {"A"}, {"C"}, {"B"})
    assert not pywhy_nx.m_separated(G, {"A"}, {"C"}, set())
    assert not pywhy_nx.m_separated(G, {"A"}, {"D"}, set())

    # check conditioning on collider of descendant in bidirected graph works
    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C", "D"])
    digraph.add_edge("B", "D")
    digraph.add_edge("A", "B")
    digraph.add_edge("C", "B")

    G = pywhy_nx.MixedEdgeGraph([digraph], ["directed"])

    assert not pywhy_nx.m_separated(G, {"A"}, {"C"}, {"D"})
    assert pywhy_nx.m_separated(G, {"A"}, {"C"}, set())

    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C", "D"])
    digraph.add_edge("B", "D")
    digraph.add_edge("A", "B")
    bigraph = nx.Graph()
    bigraph.add_edge("B", "C")
    G = pywhy_nx.MixedEdgeGraph([digraph, bigraph], ["directed", "bidirected"])

    assert not pywhy_nx.m_separated(G, {"A"}, {"C"}, {"D"})
    assert pywhy_nx.m_separated(G, {"A"}, {"C"}, set())


def test_anterior():
    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C", "D"])
    digraph.add_edge("B", "A")
    ungraph = nx.Graph()
    ungraph.add_edge("A", "D")
    ungraph.add_edge("C", "B")
    G = pywhy_nx.MixedEdgeGraph([digraph, ungraph], ["directed", "undirected"])

    result = pywhy_nx.algorithms.m_separation._anterior(G, {"A"})

    assert result == {"A", "B", "C", "D"}


def test_is_minimal_m_separator(fig5_vanderzander):

    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_1"})
    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_2"})
    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_2"}, r={"Z_1", "Z_2"})
    assert not pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", set())
    assert not pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"V_1"})
    with pytest.raises(nx.NetworkXError, match="should be no larger than proposed separating set"):
        pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_2"}, i={"Z_1", "Z_2"})
        pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"X_1"}, i={"V_1"})
    assert pywhy_nx.is_minimal_m_separator(
        fig5_vanderzander, "X", "Y", {"Z_1", "Z_2"}, i={"Z_1", "Z_2"}
    )
    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_1"}, i={"Z_1"})

    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"Z_2"}, i={"Z_2"})

    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"V_1", "Z_2"}, i={"V_1"})
    assert pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"V_1", "Z_1"}, i={"V_1"})

    assert not pywhy_nx.is_minimal_m_separator(fig5_vanderzander, "X", "Y", {"V_1"}, i={"V_1"})


def test_minimal_m_separator(fig5_vanderzander):
    # Test fork graph
    digraph = nx.DiGraph()
    digraph.add_nodes_from(["A", "B", "C"])
    digraph.add_edge("B", "A")
    digraph.add_edge("B", "C")
    G = pywhy_nx.MixedEdgeGraph([digraph], ["directed"])

    result = pywhy_nx.minimal_m_separator(G, "A", "C")
    assert result == {"B"}

    # Test undirected chain
    ungraph = nx.Graph()
    ungraph.add_nodes_from(["A", "B", "C"])
    ungraph.add_edge("B", "A")
    ungraph.add_edge("B", "C")
    G = pywhy_nx.MixedEdgeGraph([ungraph], ["undirected"])

    result = pywhy_nx.minimal_m_separator(G, "A", "C")
    assert result == {"B"}

    # Test collider is separated by empty set
    bigraph = nx.Graph()
    bigraph.add_edge("A", "B")
    bigraph.add_edge("C", "B")
    G = pywhy_nx.MixedEdgeGraph([bigraph], ["bidirected"])

    result = pywhy_nx.minimal_m_separator(G, "A", "C")
    assert result == set()

    digraph = nx.DiGraph()
    digraph.add_edge("A", "B")
    digraph.add_edge("C", "B")
    G = pywhy_nx.MixedEdgeGraph([digraph], ["directed"])

    result = pywhy_nx.minimal_m_separator(G, "A", "C")
    assert result == set()

    # Assert adjacent nodes are not m-separated by any set
    ungraph = nx.Graph()
    ungraph.add_edge("A", "B")
    G = pywhy_nx.MixedEdgeGraph([ungraph], ["undirected"])
    result = pywhy_nx.minimal_m_separator(G, "A", "B")
    assert result is None

    # Assert that mixed edge paths are handled correctly
    ungraph = nx.Graph()
    bigraph = nx.Graph()
    digraph = nx.DiGraph()
    ungraph.add_edge("A", "B")
    digraph.add_edge("B", "C")
    bigraph.add_edge("A", "D")
    digraph.add_edge("C", "D")
    digraph.add_edge("A", "E")
    ungraph.add_edge("E", "C")

    G = pywhy_nx.MixedEdgeGraph(
        [digraph, ungraph, bigraph], ["directed", "undirected", "bidirected"]
    )

    result = pywhy_nx.minimal_m_separator(G, "A", "C")

    assert result == {"E", "B"}

    # Test Fig. 5 in Van der Zander, 2019
    G = fig5_vanderzander
    result = pywhy_nx.minimal_m_separator(G, "X", "Y")
    assert result == {"Z_1"} or result == {"Z_2"}

    assert pywhy_nx.minimal_m_separator(G, "X", "Y", i={"Z_1", "Z_2"}) == {"Z_1", "Z_2"}

    assert pywhy_nx.minimal_m_separator(G, "X", "Y", i={"Z_1"}) == {"Z_1"}
    assert pywhy_nx.minimal_m_separator(G, "X", "Y", i={"Z_2"}) == {"Z_2"}
