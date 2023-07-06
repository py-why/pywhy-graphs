import networkx as nx
import pytest

from pywhy_graphs.functional.base import (
    _check_input_graph,
    add_domain_shift_function,
    add_noise_function,
    add_parent_function,
    add_soft_intervention_function,
)


def test_check_input_graph():
    # Create a valid input graph
    G = nx.DiGraph()
    G.add_node(1, exogenous_function="func1", exogenous_distribution="dist1")
    G.add_node(
        2, exogenous_function="func2", exogenous_distribution="dist2", parent_function="parent2"
    )
    G.add_node(3, exogenous_function="func3", exogenous_distribution="dist3")
    G.add_edge(1, 2)
    G.add_edge(3, 2)
    G_copy = G.copy()

    # Test a valid input graph
    _check_input_graph(G)  # No exception should be raised

    # Test an invalid input graph: not a DAG
    G.add_edge(2, 1)  # Create a cycle
    with pytest.raises(ValueError, match="The input graph must be a DAG."):
        _check_input_graph(G)

    # Test an invalid input graph: missing exogenous function
    G.remove_edge(2, 1)  # Remove the cycle
    G.nodes[1].pop("exogenous_function")
    with pytest.raises(ValueError, match="Node 1 does not have an exogenous function."):
        _check_input_graph(G)

    # Test an invalid input graph: missing exogenous distribution
    G.nodes[1]["exogenous_function"] = "func1"
    G.nodes[1].pop("exogenous_distribution")
    with pytest.raises(
        ValueError, match="Node 1 does not have an exogenous variable distribution."
    ):
        _check_input_graph(G)

    # Test an invalid input graph: missing parent function
    G = G_copy.copy()
    G.nodes[1]["exogenous_distribution"] = "dist1"
    G.nodes[2].pop("parent_function")
    with pytest.raises(
        ValueError, match="Node 2 does not have a parent function, but it has parents."
    ):
        _check_input_graph(G)

    # Test an invalid input graph: parent function without parents
    G = G_copy.copy()
    G.nodes[3]["parent_function"] = "parent3"
    G.remove_edge(3, 2)  # Remove the parent edge
    with pytest.raises(ValueError, match="Node 3 has a parent function, but it has no parents."):
        _check_input_graph(G)


def test_add_parent_function():
    """Test the add_parent_function function."""
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2)

    def parent_func(x):
        return x[0] + x[1]

    # Test adding a parent function to a node
    updated_G = add_parent_function(G, 2, parent_func)
    assert updated_G.nodes[2]["parent_function"] == parent_func

    # Test checking the 'functional' graph attribute
    assert updated_G.graph.get("functional", False) is True

    # Test checking the keyword arguments of the parent function
    with pytest.raises(ValueError, match="should have 1 arguments"):
        add_parent_function(G, 2, lambda x, y: x + y)

    # Test checking a node without any parents
    with pytest.raises(ValueError, match="should have 0 arguments"):
        add_parent_function(G, 1, parent_func)


def test_add_noise_function():
    """Test the add_noise_function function."""
    G = nx.DiGraph()
    G.add_node(1)

    def distr_func():
        return 1

    def exogenous_func(x):
        return x

    # Test adding a noise function to a node
    updated_G = add_noise_function(G, 1, distr_func, exogenous_func)
    assert updated_G.nodes[1]["exogenous_function"] == exogenous_func
    assert updated_G.nodes[1]["exogenous_distribution"] == distr_func

    # Test using the identity function when `func` is None
    updated_G = add_noise_function(G, 1, distr_func, func=None)
    assert updated_G.nodes[1]["exogenous_function"](3) == 3

    # Test checking the keyword arguments of the exogenous function
    with pytest.raises(ValueError, match="should have only 1 argument"):
        add_noise_function(G, 1, distr_func, lambda x, y: x + y)


@pytest.mark.skip()
def test_add_soft_intervention_function():
    """Test adding soft interventions."""
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2)
    G.add_edge(3, 2)

    def parent_func(x):
        return x[0] + x[1]

    # Test adding a soft intervention function to a node
    updated_G = add_soft_intervention_function(G, 2, 1, parent_func)
    assert updated_G.nodes[2]["intervention_functions"][1] == parent_func

    # Test checking the 'functional' and 'interventional' graph attributes
    assert updated_G.graph.get("functional", False) is True
    assert updated_G.graph.get("interventional", False) is True

    # Test checking the keyword arguments of the intervention function
    with pytest.raises(ValueError, match="The intervention function should take 2 argument"):
        add_soft_intervention_function(G, 2, 1, lambda x, y, z: x + y + z)

    # Test checking a node that is not a child of the specified F-node
    with pytest.raises(RuntimeError, match="Node 2 is not a child of F-node 3."):
        add_soft_intervention_function(G, 2, 3, parent_func)


@pytest.mark.skip()
def test_add_domain_shift_function():
    """Test adding domain shifts."""
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2)
    G.add_edge(3, 2)

    def parent_func(x):
        return x[0] + x[1]

    def distr_func():
        return 1

    G.nodes[2]["parent_function"] = parent_func
    G.nodes[2]["exogenous_distribution"] = distr_func

    # Test adding a domain shift function to a node
    updated_G = add_domain_shift_function(G, 2, 1, parent_func, distr_func)
    assert updated_G.nodes[2]["domain_parent_functions"][1] == parent_func
    assert updated_G.nodes[2]["domain_exogenous_distributions"][1] == distr_func

    # Test checking the 'functional' and 'multidomain' graph attributes
    assert updated_G.graph.get("functional", False) is True
    assert updated_G.graph.get("multidomain", False) is True

    # Test checking a node that is not a child of the specified S-node
    with pytest.raises(RuntimeError, match="Node 2 is not a child of S-node 3."):
        add_domain_shift_function(G, 2, 3, parent_func, distr_func)

    # Test checking when neither func nor distr_func is specified
    with pytest.raises(RuntimeError, match="Either func or distr_func must be specified."):
        add_domain_shift_function(G, 2, 1)

    # Test checking a node that does not have a parent function and func is not specified
    G.nodes[2].pop("parent_function")
    with pytest.raises(
        RuntimeError, match="Node 2 does not have a parent function and func is not specified."
    ):
        add_domain_shift_function(G, 2, 1, func=None)
