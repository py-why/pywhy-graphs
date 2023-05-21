from pywhy_graphs.algorithms import find_connected_pairs


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


def test_find_invariant_domains():
    pass


def test_add_all_snode_combinations():
    pass
