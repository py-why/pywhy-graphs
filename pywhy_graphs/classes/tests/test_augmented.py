import math

from pywhy_graphs.classes import compute_augmented_nodes


def test_compute_augmented_nodes():
    domain_indices = [1, 2, 2]
    intervention_targets = [{}, {}, {"x"}]

    # test augmented nodes
    (
        augmented_nodes,
        symmetric_diff_map,
        sigma_map,
        node_domain_map,
    ) = compute_augmented_nodes(intervention_targets, domain_indices)
    assert len(augmented_nodes) == math.comb(len(domain_indices), 2)
    assert symmetric_diff_map == {
        ("F", 2): frozenset({"x"}),
        ("F", 1): frozenset({"x"}),
        ("F", 0): frozenset(),
    }
    assert sigma_map == {("F", 2): [1, 2], ("F", 1): [0, 2], ("F", 0): [0, 1]}
    assert node_domain_map == {("F", 2): [2, 2], ("F", 1): [1, 2], ("F", 0): [1, 2]}

    domain_indices = [1, 3, 5, 2, 2]
    intervention_targets = [{}, {}, {3}, {2}, {3}]

    # test augmented nodes
    (
        augmented_nodes,
        symmetric_diff_map,
        sigma_map,
        node_domain_map,
    ) = compute_augmented_nodes(intervention_targets, domain_indices)
    assert len(augmented_nodes) == math.comb(len(domain_indices), 2)
    for node, domains in node_domain_map.items():
        assert all(domain in domain_indices for domain in domains)
        assert node in sigma_map
        assert node in symmetric_diff_map

    domain_indices = [1, 10, 10]
    intervention_targets = [{}, {}, {"x"}]
    # test augmented nodes
    (
        augmented_nodes,
        symmetric_diff_map,
        sigma_map,
        node_domain_map,
    ) = compute_augmented_nodes(intervention_targets, domain_indices)
    assert len(augmented_nodes) == math.comb(len(domain_indices), 2)
    for node, domains in node_domain_map.items():
        # the domain indices should always be part of the domain indices passed
        assert all(domain in domain_indices for domain in domains)
        assert node in sigma_map
        assert node in symmetric_diff_map
