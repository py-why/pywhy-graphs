import networkx as nx
import pytest
from scipy.stats import ttest_ind

from pywhy_graphs.functional import sample_from_graph
from pywhy_graphs.functional.additive import generate_edge_functions_for_node
from pywhy_graphs.functional.linear import (
    apply_linear_soft_intervention,
    generate_noise_for_node,
    make_random_linear_gaussian_graph,
)
from pywhy_graphs.simulate import simulate_random_er_dag


def test_make_linear_gaussian_graph():
    G = simulate_random_er_dag(n_nodes=5, seed=12345, ensure_acyclic=True)

    G = make_random_linear_gaussian_graph(G, random_state=12345)

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
        make_random_linear_gaussian_graph(G, node_mean_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="must be a list of length 2."):
        make_random_linear_gaussian_graph(G, node_std_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="must be a list of length 2."):
        make_random_linear_gaussian_graph(G, edge_weight_lims=[0], random_state=12345)

    with pytest.raises(ValueError, match="The input graph must be a DAG."):
        make_random_linear_gaussian_graph(
            nx.cycle_graph(4, create_using=nx.DiGraph), random_state=12345
        )


def test_generate_noise_for_node_works():
    G = nx.DiGraph()
    G.add_node("A")

    node_mean_lims = (0, 1)
    node_std_lims = (0, 1)

    # before adding any functionals, we cannot sample from the graph
    with pytest.raises(ValueError, match="The input graph must be a functional graph"):
        sample_from_graph(G, n_samples=1, random_state=12345)
    G.graph["functional"] = "linear_gaussian"
    with pytest.raises(ValueError, match="does not have an exogenous function"):
        sample_from_graph(G, n_samples=1, random_state=12345)

    # now when we add the exogenous function, we can sample from the graph
    G = generate_noise_for_node(G, "A", node_mean_lims, node_std_lims)

    # Check if node attributes are set properly
    assert "exogenous_distribution" in G.nodes["A"]
    assert "exogenous_function" in G.nodes["A"]

    # Check if exogenous_distribution is a callable function
    assert callable(G.nodes["A"]["exogenous_distribution"])

    # Check if exogenous_function is a callable function
    assert callable(G.nodes["A"]["exogenous_function"])

    # sample from the graph should work, since there is only one node
    sample_from_graph(G, n_samples=1, random_state=12345)


def test_apply_linear_soft_intervention():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")

    node_mean_lims = (0, 1)
    node_std_lims = (0, 1)

    G = generate_noise_for_node(G, "A", node_mean_lims, node_std_lims)
    G = generate_noise_for_node(G, "B", node_mean_lims, node_std_lims)
    G = generate_edge_functions_for_node(G, "B", random_state=1234)
    G.graph["functional"] = "linear_gaussian"
    targets = {"B"}

    # Before intervening, the functions are the same
    for target in targets:
        assert (
            G.nodes[target]["exogenous_distribution"]
            == G.copy().nodes[target]["exogenous_distribution"]
        )

    # Test additive intervention type
    G_intervened = apply_linear_soft_intervention(
        G.copy(), targets, intervention_value=5.0, type="additive"
    )

    # Check if the target nodes have modified exogenous_distribution functions
    for target in targets:
        assert (
            G.nodes[target]["exogenous_distribution"]
            != G_intervened.nodes[target]["exogenous_distribution"]
        )

    # Check if the exogenous_distribution functions of non-target nodes remain unchanged
    non_target_nodes = set(G.nodes) - targets
    for node in non_target_nodes:
        assert (
            G.nodes[node]["exogenous_distribution"]
            == G_intervened.nodes[node]["exogenous_distribution"]
        )

    # now ensure that the two distributions are different and the same where they should be
    # the node A is not intervened on, so the distributions should be the same
    # while the node B is intervened on, so the distributions should be different
    df_original = sample_from_graph(G, n_samples=1000, random_state=12345)
    df_intervened = sample_from_graph(G_intervened, n_samples=1000, random_state=12345)
    _, pvalue = ttest_ind(df_original["A"], df_intervened["A"])
    assert pvalue > 0.05
    _, pvalue = ttest_ind(df_original["B"], df_intervened["B"])
    assert pvalue < 0.05


def test_apply_linear_soft_intervention_errors():
    targets = {"A", "B"}

    # Test intervention on a non-linear Gaussian graph
    H = nx.DiGraph()
    H.add_node("X")
    H.add_node("Y")
    H.add_edge("X", "Y")

    with pytest.raises(ValueError, match="The input graph must be a linear Gaussian graph."):
        H.graph["linear_gaussian"] = False
        apply_linear_soft_intervention(H, targets, type="additive")

    with pytest.raises(ValueError, match="All targets"):
        H.graph["linear_gaussian"] = True
        apply_linear_soft_intervention(H, {1, 2}, type="additive")
