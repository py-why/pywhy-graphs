import pytest

from pywhy_graphs import AugmentedGraph
from pywhy_graphs.algorithms import (
    add_all_snode_combinations,
    compute_invariant_domains_per_node,
    find_connected_pairs,
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


@pytest.mark.skip()
def test_compute_invariant_domains_per_node():
    # Create a test graph
    G = AugmentedGraph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4)], edge_type="directed")

    # Call the function to compute invariant domains for node 1
    G = compute_invariant_domains_per_node(G, node=1)

    # Assert that the computed invariant domains for node 1 are correct
    assert G.nodes[1]["invariant_domains"] == {1, 2, 3}

    # Call the function again to compute invariant domains for node 2
    G = compute_invariant_domains_per_node(G, node=2)

    # Assert that the computed invariant domains for node 2 are correct
    assert G.nodes[2]["invariant_domains"] == {1, 2, 3}

    # Call the function to compute invariant domains for node 3
    G = compute_invariant_domains_per_node(G, node=3)

    # Assert that the computed invariant domains for node 3 are correct
    assert G.nodes[3]["invariant_domains"] == {1, 2, 3}

    # Call the function to compute invariant domains for node 4 (no S-nodes, no change)
    G = compute_invariant_domains_per_node(G, node=4)

    # Assert that the invariant domains for node 4 remain empty (no S-nodes added)
    assert G.nodes[4]["invariant_domains"] == set()

    # Test the inconsistency handling
    with pytest.warns(UserWarning):
        G = compute_invariant_domains_per_node(G, node=1, inconsistency="warn")

    with pytest.raises(RuntimeError):
        G = compute_invariant_domains_per_node(G, node=2, inconsistency="raise")


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
