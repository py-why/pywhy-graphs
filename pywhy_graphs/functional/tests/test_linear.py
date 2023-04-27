import networkx as nx

from pywhy_graphs.functional import make_graph_linear_gaussian
from pywhy_graphs.simulate import simulate_random_er_dag


def test_make_linear_gaussian_graph():
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    G = make_graph_linear_gaussian(G, random_state=12345)

    assert all(key in nx.get_node_attributes(G, "parent_functions") for key in G.nodes)
    assert all(key in nx.get_node_attributes(G, "gaussian_noise_function") for key in G.nodes)
