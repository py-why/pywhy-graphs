import networkx as nx
import pytest

from pywhy_graphs.functional import make_graph_linear_gaussian, make_graph_multidomain
from pywhy_graphs.simulate import simulate_random_er_dag


@pytest.mark.parametrize("n_domains", [2])
def test_make_linear_gaussian_graph(n_domains):
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    # make linear graph SCM
    G = make_graph_linear_gaussian(G, random_state=12345)

    # make multidomain SCM
    G = make_graph_multidomain(G, n_domains=n_domains, random_state=12345)

    s_nodes = G.graph["S-nodes"]

    assert len(s_nodes) == n_domains - 1

    assert G.nodes(data=True)[s_nodes[0]] == {"domain_ids": (1, 2)}
    assert all(
        key in nx.get_node_attributes(G, "parent_functions")
        for key in G.nodes
        if key not in s_nodes
    )
    assert all(
        key in nx.get_node_attributes(G, "gaussian_noise_function")
        for key in G.nodes
        if key not in s_nodes
    )
