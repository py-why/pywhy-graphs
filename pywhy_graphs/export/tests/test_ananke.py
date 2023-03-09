import ananke
import networkx as nx
import pytest

import pywhy_graphs
from ananke.graphs import Graph, ADMG, CG, DAG
import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.export import ananke_to_graph, graph_to_ananke


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
    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "C"), ("B", "D")]
    ud_edges = [("B", "A"), ("C", "D")]
    graph = CG(vertices=vertices, di_edges=di_edges, ud_edges=ud_edges)

    directed_edges = nx.DiGraph(di_edges)
    undirected_edges = nx.Graph(ud_edges)

    expected_graph = pywhy_nx.MixedEdgeGraph()
    expected_graph.add_nodes_from(vertices)
    expected_graph.add_edge_type(directed_edges, "directed")
    expected_graph.add_edge_type(undirected_edges, "undirected")

    return graph, expected_graph


@pytest.mark.parametrize(
    "ananke_G, expected_G, graph_type",
    [
        [*dag(), "dag"],
        [*admg(), "admg"],
        [*cg(), "cg"],
    ],
)
def test_graph_to_ananke_roundtrip(
    ananke_G: Graph, expected_G: pywhy_nx.MixedEdgeGraph, graph_type: str
):
    # Convert to Ananke graph
    result = graph_to_ananke(expected_G)

    # Convert back to MixedEdgeGraph
    graph = ananke_to_graph(result)

    for edge_type, subG in graph.get_graphs().items():
        assert edge_type in expected_G.edge_types
        assert nx.is_isomorphic(subG, expected_G.get_graphs(edge_type))
