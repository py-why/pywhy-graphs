import pytest

from pywhy_graphs import AugmentedGraph
from pywhy_graphs.algorithms import (
    add_all_snode_combinations,
    compute_invariant_domains_per_node,
    find_connected_pairs,
    remove_snode_edge,
)


def test_find_connected_domain_pairs():
    # Example usage:
    tuples = [(1, 2), (2, 3), (1, 3)]
    max_number = 4
    connected_pairs = find_connected_pairs(tuples, max_number)
    assert connected_pairs == {(2, 3), (1, 2), (1, 3)}

    tuples = [(1, 2), (2, 3)]
    max_number = 4
    connected_pairs = find_connected_pairs(tuples, max_number)
    assert connected_pairs == {(2, 3), (1, 2), (1, 3)}

    max_number = 5
    connected_pairs = find_connected_pairs(tuples, max_number)
    assert connected_pairs == {(2, 3), (1, 2), (1, 3)}

    tuples = [(1, 2), (2, 3), (2, 4)]
    connected_pairs = find_connected_pairs(tuples, max_number)
    assert connected_pairs == {(1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4)}


def test_add_all_snode_combinations():
    # Create a test graph
    G = AugmentedGraph()
    G.add_nodes_from([1])

    # Call the function to add S-nodes with n_domains = 3
    G, s_node_domains = add_all_snode_combinations(G, n_domains=3)

    # Assert that the number of S-nodes added matches the expected value (3 choose 2 = 3)
    assert len(s_node_domains) == 3

    # Assert that each S-node has the correct domain_ids attribute
    assert G.nodes[s_node_domains[(1, 2)]]["domain_ids"] == (1, 2)
    assert G.nodes[s_node_domains[(1, 3)]]["domain_ids"] == (1, 3)
    assert G.nodes[s_node_domains[(2, 3)]]["domain_ids"] == (2, 3)

    # Assert that the edges are added
    assert len(list(G.edges())) == len(s_node_domains)


def example_augmented_graph(n_domains=3):
    # Create an example AugmentedGraph for testing
    G = AugmentedGraph()
    G.add_node("x")

    G, _ = add_all_snode_combinations(G, n_domains=n_domains)
    for snode in G.s_nodes:
        G.add_edge(snode, "x")
    return G


def test_compute_invariant_domains_per_node_when_three_domains():
    n_domains = 3
    G = example_augmented_graph(n_domains=n_domains)

    # Compute the invariant domains for node "A" with 3 domains
    # which should be none when the S-nodes are fully connected
    G_result = compute_invariant_domains_per_node(G, "x", n_domains=n_domains)
    assert G_result.nodes()["x"]["invariant_domains"] == set()

    # removing one S-node should only result in two pairwise invariant domains
    snode = ("S", 0)
    remove_snode_edge(G, snode, "x")
    G_result = compute_invariant_domains_per_node(G, "x", n_domains=n_domains)
    assert G_result.nodes()["x"]["invariant_domains"] == set(G.s_node_domain_ids[snode])

    # if we remove another S-node without preserving the invariance, it should be caught
    snode = ("S", 1)
    G_copy = remove_snode_edge(G.copy(), snode, "x", preserve_invariance=False)
    with pytest.raises(RuntimeError, match="Inconsistency in S-nodes"):
        G_result = compute_invariant_domains_per_node(G_copy, "x", n_domains=n_domains)

    # removing another S-node should result in all domains being invariant
    G_copy = remove_snode_edge(G.copy(), snode, "x", preserve_invariance=True)
    G_result = compute_invariant_domains_per_node(G_copy, "x", n_domains=n_domains)
    assert G_result.nodes()["x"]["invariant_domains"] == set(G.domain_ids)

    for snode in G.s_nodes:
        assert not G_result.has_edge(snode, "x")


def test_compute_invariant_domains_per_node_when_many_domains():
    n_domains = 4
    G = example_augmented_graph(n_domains=n_domains)

    # Compute the invariant domains for node "A" with 3 domains
    # which should be none when the S-nodes are fully connected
    G_result = compute_invariant_domains_per_node(G, "x", n_domains=n_domains)
    assert G_result.nodes()["x"]["invariant_domains"] == set()

    # map domain IDs
    snode_domains = G.domain_ids_to_snodes

    # removing one S-node should only result in two pairwise invariant domains
    snode = snode_domains[(1, 2)]
    remove_snode_edge(G, snode, "x")
    snode = snode_domains[(2, 4)]
    remove_snode_edge(G, snode, "x")

    G_result = compute_invariant_domains_per_node(G, "x", n_domains=n_domains)
    assert G_result.nodes()["x"]["invariant_domains"] == set([1, 2, 4])
    for snode in G.s_nodes:
        if all(domain in [1, 2, 4] for domain in G.s_node_domain_ids[snode]):
            assert not G_result.has_edge(snode, "x")
        else:
            assert G_result.has_edge(snode, "x")
