import networkx as nx
import pytest

from pywhy_graphs.functional import make_graph_linear_gaussian, sample_from_graph
from pywhy_graphs.simulate import simulate_random_er_dag


def test_make_linear_gaussian_graph():
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    G = make_graph_linear_gaussian(G, random_state=12345)

    assert all(
        key in nx.get_node_attributes(G, "parent_function")
        for key in G.nodes
        if len(list(G.predecessors(key))) > 0
    )
    assert all(key in nx.get_node_attributes(G, "exogenous_function") for key in G.nodes)

    # sample from the graph should work
    df = sample_from_graph(G, n_samples=2, random_state=12345)
    assert df.shape == (2, len(G.nodes))


def test_make_linear_gaussian_graph_errors():
    G = simulate_random_er_dag(n_nodes=2, seed=12345, ensure_acyclic=True)

    with pytest.raises(ValueError, match="must be a list of length 2."):
        make_graph_linear_gaussian(G, node_mean_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="must be a list of length 2."):
        make_graph_linear_gaussian(G, node_std_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="must be a list of length 2."):
        make_graph_linear_gaussian(G, edge_weight_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="The input graph must be a DAG."):
        make_graph_linear_gaussian(nx.cycle_graph(4, create_using=nx.DiGraph), random_state=12345)
