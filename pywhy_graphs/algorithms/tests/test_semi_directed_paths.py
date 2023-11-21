import networkx as nx
import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.algorithms import all_semi_directed_paths, is_semi_directed_path


# Fixture to create a sample mixed-edge graph for testing
@pytest.fixture
def sample_mixed_edge_graph():
    directed_G = nx.DiGraph([("X", "Y"), ("Z", "X")])
    bidirected_G = nx.Graph([("X", "Y")])
    directed_G.add_nodes_from(bidirected_G.nodes)
    bidirected_G.add_nodes_from(directed_G.nodes)
    G = pywhy_nx.MixedEdgeGraph(
        graphs=[directed_G, bidirected_G], edge_types=["directed", "bidirected"], name="IV Graph"
    )
    return G


def test_empty_path_not_semi_directed(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    assert not is_semi_directed_path(G, [])


def test_single_node_path(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    assert is_semi_directed_path(G, ["X"])


def test_nonexistent_node_path(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    assert not is_semi_directed_path(G, ["A", "B"])


def test_repeated_nodes_path(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    assert not is_semi_directed_path(G, ["X", "Y", "X"])


def test_valid_semi_directed_path(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    G.add_edge("A", "Z", edge_type="directed")
    G.add_edge_type(nx.DiGraph(), "circle")
    G.add_edge("Z", "A", edge_type="circle")
    assert is_semi_directed_path(G, ["Z", "X"])
    assert is_semi_directed_path(G, ["A", "Z", "X"])


def test_invalid_semi_directed_path(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    assert not is_semi_directed_path(G, ["Y", "X"])

    # there is a bidirected edge between X and Y
    assert not is_semi_directed_path(G, ["X", "Y"])
    assert not is_semi_directed_path(G, ["Z", "X", "Y"])


def test_empty_paths(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "A"
    target = "B"
    with pytest.raises(nx.NodeNotFound, match="source node A not in graph"):
        all_semi_directed_paths(G, source, target)

    G.add_node(source)
    G.add_node(target)
    paths = all_semi_directed_paths(G, source, target)
    assert list(paths) == []


def test_no_paths(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "Y"
    target = "X"
    cutoff = 3
    paths = all_semi_directed_paths(G, source, target, cutoff)
    assert list(paths) == []


def test_multiple_paths(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    G.add_edge_type(nx.DiGraph(), "circle")
    G.add_edge("A", "Z", edge_type="directed")
    G.add_edge("A", "B", edge_type="circle")
    G.add_edge("B", "A", edge_type="circle")
    G.add_edge("B", "Z", edge_type="circle")

    source = "A"
    target = "X"
    cutoff = 3
    paths = all_semi_directed_paths(G, source, target, cutoff)
    paths = list(paths)
    assert len(paths) == 2
    assert all(path in paths for path in [["A", "Z", "X"], ["A", "B", "Z", "X"]])


def test_long_cutoff(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "Z"
    target = "X"
    cutoff = 10  # Cutoff longer than the actual path length
    paths = all_semi_directed_paths(G, source, target, cutoff)
    assert list(paths) == [[source, target]]
