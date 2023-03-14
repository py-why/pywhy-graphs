import networkx as nx
import pytest
from ananke.graphs import ADMG, BG, CG, DAG, SG, UG, Graph

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.export import ananke_to_graph, graph_to_ananke
from pywhy_graphs.testing import assert_mixed_edge_graphs_isomorphic


def dag():

    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
    graph = DAG(vertices=vertices, di_edges=di_edges)
    expected_graph = pywhy_graphs.ADMG(di_edges, bi_edges=[])

    return graph, expected_graph


def admg():

    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
    bi_edges = [("A", "C"), ("B", "D")]
    graph = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)

    directed_edges = nx.DiGraph(di_edges)
    bidirected_edges = nx.Graph(bi_edges)
    expected_graph = pywhy_graphs.ADMG(directed_edges, bidirected_edges)
    expected_graph.add_nodes_from(vertices)

    return graph, expected_graph


def cg():
    # No partially directed cycles allowed, currently using CPDAG to represent
    # a chain graph
    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "C"), ("B", "D")]
    ud_edges = [("B", "A"), ("C", "D")]
    graph = CG(vertices=vertices, di_edges=di_edges, ud_edges=ud_edges)

    directed_edges = nx.DiGraph(di_edges)
    undirected_edges = nx.Graph(ud_edges)

    expected_graph = pywhy_graphs.CPDAG(directed_edges, undirected_edges)
    expected_graph.add_nodes_from(vertices)

    return graph, expected_graph


def ug():
    vertices = ["A", "B", "C", "D"]
    ud_edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    graph = UG(vertices, ud_edges=ud_edges)

    undirected_edges = nx.Graph(ud_edges)
    expected_graph = pywhy_nx.MixedEdgeGraph()
    expected_graph.add_nodes_from(vertices)
    expected_graph.add_edge_type(undirected_edges, "undirected")

    return graph, expected_graph


def bg():
    vertices = ["A", "B", "C", "D"]
    bi_edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    graph = BG(vertices, bi_edges=bi_edges)

    bidirected_edges = nx.Graph(bi_edges)
    expected_graph = pywhy_nx.MixedEdgeGraph()
    expected_graph.add_nodes_from(vertices)
    expected_graph.add_edge_type(bidirected_edges, "bidirected")

    return graph, expected_graph


def sg():
    # Segregated graph needs to satisfy the property that undirected and
    # bidirected edges never meet, and that undirected edges can't be oriented
    # to form a directed cycle
    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "C"), ("B", "D")]
    bi_edges = [("A", "B")]
    ud_edges = [("C", "D")]
    graph = SG(vertices=vertices, di_edges=di_edges, ud_edges=ud_edges, bi_edges=bi_edges)

    directed_edges = nx.DiGraph(di_edges)
    undirected_edges = nx.Graph(ud_edges)
    bidirected_edges = nx.Graph(bi_edges)

    expected_graph = pywhy_nx.MixedEdgeGraph()
    expected_graph.add_nodes_from(vertices)
    expected_graph.add_edge_type(directed_edges, "directed")
    expected_graph.add_edge_type(undirected_edges, "undirected")
    expected_graph.add_edge_type(bidirected_edges, "bidirected")

    return graph, expected_graph


@pytest.mark.parametrize(
    "ananke_G, expected_G, graph_type",
    [
        [*dag(), "dag"],
        [*admg(), "admg"],
        [*cg(), "cg"],
        [*ug(), "ug"],
        [*bg(), "bg"],
        [*sg(), "sg"],
    ],
)
def test_graph_to_ananke_roundtrip(
    ananke_G: Graph, expected_G: pywhy_nx.MixedEdgeGraph, graph_type: str
):
    # Convert to Ananke graph
    result = graph_to_ananke(expected_G)

    # Convert back to MixedEdgeGraph
    graph = ananke_to_graph(result)

    assert assert_mixed_edge_graphs_isomorphic(graph, expected_G)
