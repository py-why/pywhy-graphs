# import math

# import networkx as nx
import pytest

from pywhy_graphs.functional import make_graph_linear_gaussian
from pywhy_graphs.simulate import simulate_random_er_dag


@pytest.mark.parametrize("n_domains, n_invariances_to_try", [[2, 0], [3, 1], [5, 2]])
def test_make_linear_gaussian_graph(n_domains, n_invariances_to_try):
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    # make linear graph SCM
    G = make_graph_linear_gaussian(G, random_state=12345)

    # make multidomain SCM
    # G = make_graph_multidomain(
    #     G, n_domains=n_domains, n_invariances_to_try=n_invariances_to_try, random_state=12345
    # )

    # check the number of S-nodes created is correct
    # s_nodes = G.graph["S-nodes"]
    # assert len(s_nodes) == math.comb(n_domains, 2)

    # # check the S-nodes contain domain IDs
    # assert G.nodes(data=True)[s_nodes[0]] == {"domain_ids": (1, 2)}

    # assert all(
    #     key in nx.get_node_attributes(G, "parent_functions")
    #     for key in G.nodes
    #     if key not in s_nodes
    # )
    # assert all(
    #     key in nx.get_node_attributes(G, "gaussian_noise_function")
    #     for key in G.nodes
    #     if key not in s_nodes
    # )

    # # for each child of a S-node, they should have a domain_gaussian_noise_function
    # # dictating how to sample the noise function per variant domain
    # for node in G.nodes:
    #     if any(G.has_edge(s_node, node) for s_node in s_nodes):
    #         assert "domain_gaussian_noise_function" in G.nodes[node]
    #         assert "invariant_domains" in G.nodes[node]
    #     else:
    #         assert "domain_gaussian_noise_function" not in G.nodes[node]
    #         assert "invariant_domains" not in G.nodes[node]
