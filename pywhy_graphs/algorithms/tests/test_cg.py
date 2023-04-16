import pytest

from pywhy_graphs import CG
from pywhy_graphs.algorithms import is_valid_cg


@pytest.fixture
def cg_simple_partially_directed_cycle():
    graph = CG()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edge("A", "B", graph.directed_edge_name)
    graph.add_edge("D", "C", graph.directed_edge_name)
    graph.add_edge("B", "D", graph.undirected_edge_name)
    graph.add_edge("A", "C", graph.undirected_edge_name)

    return graph


@pytest.fixture
def cg_multiple_blocks_partially_directed_cycle():

    graph = CG()
    graph.add_nodes_from(["A", "B", "C", "D", "E", "F", "G"])
    graph.add_edge("A", "B", graph.directed_edge_name)
    graph.add_edge("D", "C", graph.directed_edge_name)
    graph.add_edge("B", "D", graph.undirected_edge_name)
    graph.add_edge("A", "C", graph.undirected_edge_name)
    graph.add_edge("E", "F", graph.undirected_edge_name)
    graph.add_edge("F", "G", graph.undirected_edge_name)
    graph.add_edge("G", "E", graph.undirected_edge_name)

    return graph


@pytest.fixture
def square_graph():
    graph = CG()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edge("A", "B", graph.undirected_edge_name)
    graph.add_edge("B", "C", graph.undirected_edge_name)
    graph.add_edge("C", "D", graph.undirected_edge_name)
    graph.add_edge("C", "A", graph.undirected_edge_name)

    return graph


@pytest.fixture
def fig_g1_frydenberg():
    graph = CG()
    graph.add_nodes_from(["a", "b", "g", "m", "d"])
    graph.add_edge("a", "b", graph.undirected_edge_name)
    graph.add_edge("b", "g", graph.directed_edge_name)
    graph.add_edge("g", "d", graph.undirected_edge_name)
    graph.add_edge("d", "m", graph.undirected_edge_name)
    graph.add_edge("a", "m", graph.directed_edge_name)

    return graph


@pytest.fixture
def fig_g2_frydenberg():
    graph = CG()
    graph.add_nodes_from(["b", "g", "d", "m", "a"])
    graph.add_edge("a", "m", graph.directed_edge_name)
    graph.add_edge("m", "g", graph.undirected_edge_name)
    graph.add_edge("m", "d", graph.directed_edge_name)
    graph.add_edge("g", "d", graph.directed_edge_name)
    graph.add_edge("b", "g", graph.directed_edge_name)

    return graph


@pytest.fixture
def fig_g3_frydenberg():
    graph = CG()
    graph.add_nodes_from(["a", "b", "g"])
    graph.add_edge("b", "a", graph.undirected_edge_name)
    graph.add_edge("a", "g", graph.undirected_edge_name)
    graph.add_edge("b", "g", graph.directed_edge_name)

    return graph


@pytest.fixture
def fig_g4_frydenberg():
    graph = CG()
    graph.add_nodes_from(["b", "g", "d", "m", "a"])
    graph.add_edge("b", "g", graph.directed_edge_name)
    graph.add_edge("a", "b", graph.undirected_edge_name)
    graph.add_edge("g", "d", graph.undirected_edge_name)
    graph.add_edge("d", "m", graph.undirected_edge_name)
    graph.add_edge("m", "a", graph.undirected_edge_name)
    graph.add_edge("a", "g", graph.directed_edge_name)

    return graph


@pytest.mark.parametrize(
    "G",
    [
        "cg_simple_partially_directed_cycle",
        "cg_multiple_blocks_partially_directed_cycle",
        "fig_g3_frydenberg",
        "fig_g4_frydenberg",
    ],
)
def test_graphs_are_not_valid_cg(G, request):
    graph = request.getfixturevalue(G)

    assert not is_valid_cg(graph)


@pytest.mark.parametrize(
    "G",
    [
        "square_graph",
        "fig_g1_frydenberg",
        "fig_g2_frydenberg",
    ],
)
def test_graphs_are_valid_cg(G, request):
    graph = request.getfixturevalue(G)

    assert is_valid_cg(graph)
