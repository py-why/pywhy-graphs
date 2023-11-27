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

    G.add_edge_type(nx.DiGraph(), "circle")
    G.add_edge("A", "Z", edge_type="directed")
    G.add_edge("Z", "A", edge_type="circle")
    G.add_edge("A", "B", edge_type="circle")
    G.add_edge("B", "A", edge_type="circle")
    G.add_edge("B", "Z", edge_type="circle")
    return G


class TestIsSemiDirectedPath:
    def test_empty_path_not_semi_directed(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert not is_semi_directed_path(G, [])

    def test_single_node_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert is_semi_directed_path(G, ["X"])

    def test_nonexistent_node_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert not is_semi_directed_path(G, ["1", "2"])

    def test_repeated_nodes_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert not is_semi_directed_path(G, ["X", "Y", "X"])

    def test_non_connected_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert not is_semi_directed_path(G, ["A", "X"])

    def test_valid_semi_directed_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert is_semi_directed_path(G, ["Z", "X"])
        assert is_semi_directed_path(G, ["A", "Z", "X"])

    def test_invalid_semi_directed_path(self, sample_mixed_edge_graph):
        G = sample_mixed_edge_graph
        assert not is_semi_directed_path(G, ["Y", "X"])

        # there is a bidirected edge between X and Y
        assert not is_semi_directed_path(G, ["X", "Y"])
        assert not is_semi_directed_path(G, ["Z", "X", "Y"])


def test_node_not_in_graph():
    G = nx.Graph()
    G.add_edge("X", "Y")
    with pytest.raises(nx.NodeNotFound):
        all_semi_directed_paths(G, "A", "X")

    with pytest.raises(nx.NodeNotFound):
        all_semi_directed_paths(G, "X", 1)


def test_target_is_single_node_in_graph(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "X"
    paths = all_semi_directed_paths(G, source, "Y")
    assert list(paths) == []


def test_source_same_as_target(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "X"
    paths = all_semi_directed_paths(G, source, source)
    assert list(paths) == []


def test_cutoff_none(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "Z"
    paths = all_semi_directed_paths(G, source, "X", cutoff=None)
    assert list(paths) == [["Z", "X"]]


def test_cutoff_less_than_one(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "X"
    paths = all_semi_directed_paths(G, source, "Y", cutoff=0)
    assert list(paths) == []


def test_empty_paths(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "1"
    target = "B"
    with pytest.raises(nx.NodeNotFound, match=f"source node {source} not in graph"):
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

    source = "A"
    target = "X"
    cutoff = 3
    paths = all_semi_directed_paths(G, source, target, cutoff)
    paths = list(paths)

    dig = nx.path_graph(5, create_using=nx.DiGraph())
    G.add_edges_from(dig.edges(), edge_type="directed")
    G.add_edge("A", 0, edge_type="circle")

    assert len(paths) == 2
    assert all(path in paths for path in [["A", "Z", "X"], ["A", "B", "Z", "X"]])

    # for a short cutoff, there is only one path
    cutoff = 2
    paths = all_semi_directed_paths(G, source, target, cutoff)
    assert all(path in paths for path in [["A", "Z", "X"]])

    # for an even shorter cutoff, there are no paths now
    cutoff = 1
    paths = all_semi_directed_paths(G, source, target, cutoff)
    assert list(paths) == []


def test_long_cutoff(sample_mixed_edge_graph):
    G = sample_mixed_edge_graph
    source = "Z"
    target = "X"
    cutoff = 10  # Cutoff longer than the actual path length
    print(G.edges())
    paths = all_semi_directed_paths(G, source, target, cutoff)
    assert list(paths) == [[source, target]]
