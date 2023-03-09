from ananke.graphs import ADMG, BG, CG, DAG, SG, UG
from ananke.graphs.graph import Graph
import pytest
import networkx as nx

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.export import graph_to_ananke, ananke_to_graph


def dag():

    vertices = ["A", "B", "C", "D"]
    di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
    graph = DAG(vertices=vertices, di_edges=di_edges)
    expected_graph = nx.DiGraph(di_edges)

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


def test_graph_to_ananke_dag_roundtrip():
    ananke_G, expected_G = dag()

    graph = ananke_to_graph(ananke_G)

    assert nx.is_isomorphic(graph, expected_G)


@pytest.mark.parametrize(
    "ananke_G, expected_G, graph_type",
    [
        [*admg(), "admg"],
        # [*dag(), "dag"],
        # [*cg(), "cg"],
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
