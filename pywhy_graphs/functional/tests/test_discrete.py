import networkx as nx
import numpy as np
import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD
from scipy.stats import chi2_contingency

from pywhy_graphs.functional.base import sample_from_graph
from pywhy_graphs.functional.discrete import add_cpd_for_node, make_random_discrete_graph


def test_add_cpd_for_node():
    # Create a test graph
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    G_copy = G.copy()

    # Define the CPD for node 'A' (no parents)
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.3], [0.7]])

    # Test adding CPD for node 'A' (no parents)
    G = add_cpd_for_node(G, "A", cpd_a)
    assert "cpd" in G.nodes["A"]
    assert "cardinality" in G.nodes["A"]
    assert "possible_values" in G.nodes["A"]
    assert G.nodes["A"]["possible_values"] == [0, 1]
    assert G.nodes["A"]["cardinality"] == 2
    assert G.graph["functional"] == "discrete"

    # Define the CPD for node 'B' (with parent 'A')
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.2, 0.8], [0.6, 0.4]],
        evidence=["A"],
        evidence_card=[2],
    )

    # Test adding CPD for node 'B' without adding parent 'A' first
    with pytest.raises(RuntimeError, match="CPD for parent A of node B must be defined first."):
        add_cpd_for_node(G_copy, "B", cpd_b)

    # Test adding CPD for node 'B' after adding parent 'A'
    G = add_cpd_for_node(G, "B", cpd_b)
    assert "cpd" in G.nodes["B"]
    assert "cardinality" in G.nodes["B"]
    assert "possible_values" in G.nodes["B"]
    assert G.graph["functional"] == "discrete"

    # Test adding CPD for node 'B' with existing CPD (overwrite=False)
    with pytest.raises(RuntimeError, match="A CPD exists in G for"):
        add_cpd_for_node(G, "B", cpd_b)

    # Test adding CPD for node 'B' with existing CPD (overwrite=True)
    G = add_cpd_for_node(G, "B", cpd_b, overwrite=True)
    assert "cpd" in G.nodes["B"]
    assert "cardinality" in G.nodes["B"]
    assert "possible_values" in G.nodes["B"]
    assert G.graph["functional"] == "discrete"

    # Test adding CPD for node 'B' with invalid CPD
    invalid_cpd = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.2, 0.8], [0.6, 0.4]],
        evidence=["C"],
        evidence_card=[2],
    )
    with pytest.raises(ValueError):
        add_cpd_for_node(G, "B", invalid_cpd, overwrite=True)

    # Test adding CPD for node 'B' with valid CPD
    cpd_b_valid = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.1, 0.9], [0.3, 0.7]],
        evidence=["A"],
        evidence_card=[2],
    )
    G = add_cpd_for_node(G, "B", cpd_b_valid, overwrite=True)
    assert "cpd" in G.nodes["B"]
    assert "cardinality" in G.nodes["B"]
    assert "possible_values" in G.nodes["B"]
    assert G.graph["functional"] == "discrete"

    # Test adding CPD for node 'B' with different variable cardinality
    cpd_b_diff_cardinality = TabularCPD(
        variable="B",
        variable_card=3,
        values=[[0.1, 0.3], [0.2, 0.4], [0.7, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    G = add_cpd_for_node(G, "B", cpd_b_diff_cardinality, overwrite=True)
    assert G.nodes["B"]["cardinality"] == 3

    # Test adding CPD for node 'B' with different variable cardinality
    cpd_b_diff_cardinality = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
        evidence=["A"],
        evidence_card=[3],
    )
    with pytest.raises(RuntimeError, match="The cardinality of parent variable A"):
        G = add_cpd_for_node(G, "B", cpd_b_diff_cardinality, overwrite=True)
    assert G.nodes["B"]["cardinality"] == 3

    # Test adding CPD for node 'B' with different variable cardinality and parent cardinality
    cpd_b_diff_cardinality_parent = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.1, 0.9], [0.3, 0.7]],
        evidence=["A"],
        evidence_card=[2],
    )
    add_cpd_for_node(G, "B", cpd_b_diff_cardinality_parent, overwrite=True)
    assert G.nodes["B"]["cardinality"] == 2


def test_sample_from_discrete_cpd_graph():
    # Create a test graph
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    G.add_node("C")

    # Define the CPD for node 'A' (no parents)
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.25], [0.75]])
    G = add_cpd_for_node(G, "A", cpd_a)

    # Define the CPD for node 'C' (no parents)
    cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.2], [0.7], [0.1]])
    G = add_cpd_for_node(G, "C", cpd_c)

    # Define the CPD for node 'B' (with parent 'A')
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.1, 0.9], [0.3, 0.7]],
        evidence=["A"],
        evidence_card=[2],
    )
    G = add_cpd_for_node(G, "B", cpd_b)

    # Test sampling from graph
    df = sample_from_graph(G, n_samples=2000, n_jobs=1, random_state=0)
    assert df.shape == (2000, 3)
    assert set(df["A"].unique().tolist()) == set([0, 1])
    assert set(df["C"].unique().tolist()) == set([0, 1, 2])
    assert set(df["B"].unique().tolist()) == set([0, 1])

    # Chi-square test of independence where variables are independent
    contingency_table = pd.crosstab(df["A"], df["C"])
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p > 0.05

    # Chi-square test of independence where variables are independent
    contingency_table = pd.crosstab(df["B"], df["C"])
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p > 0.05

    # setup contingency table
    contingency_table = pd.crosstab(df["A"], df["B"])
    # Chi-square test of independence where variables are dependent
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p < 0.05


def test_make_random_discrete_graph():
    G = nx.DiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from([("A", "B"), ("A", "C")])
    G.add_edge("D", "B")

    cardinality_lims = [2, 3]
    weight_lims = [1, 10]
    noise_ratio_lims = [0.0, 0.0]

    rng = np.random.default_rng(42)  # Set a specific random seed for reproducibility

    altered_G = make_random_discrete_graph(
        G,
        cardinality_lims=cardinality_lims,
        weight_lims=weight_lims,
        noise_ratio_lims=noise_ratio_lims,
        random_state=rng,
    )

    # structure of the graph should stay the same
    assert set(altered_G.nodes) == {"A", "B", "C", "D"}
    assert altered_G.has_edge("A", "B")
    assert altered_G.has_edge("A", "C")
    assert altered_G.has_edge("D", "B")

    for node in altered_G.nodes:
        assert (
            altered_G.nodes[node]["cardinality"] >= cardinality_lims[0]
            and altered_G.nodes[node]["cardinality"] <= cardinality_lims[1]
        )

        assert isinstance(altered_G.nodes[node]["cpd"], TabularCPD)
        assert (
            len(altered_G.nodes[node]["cpd"].state_names[node])
            == altered_G.nodes[node]["cardinality"]
        )

        if node == "B":
            assert len(altered_G.nodes[node]["cpd"].state_names["A"]) == 2
        elif node == "C":
            assert len(altered_G.nodes[node]["cpd"].state_names["A"]) == 2

        if node not in ("A", "D"):
            assert altered_G.nodes[node]["noise_ratio"] >= noise_ratio_lims[0]
            assert altered_G.nodes[node]["noise_ratio"] <= noise_ratio_lims[1]
        else:
            assert altered_G.nodes[node]["noise_ratio"] == 1.0

    df = sample_from_graph(altered_G, n_samples=2000, n_jobs=1, random_state=0)
    # Chi-square test of independence where variables are independent
    contingency_table = pd.crosstab(df["A"], df["C"])
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p < 0.05

    # Chi-square test of independence where variables are independent
    contingency_table = pd.crosstab(df["A"], df["B"])
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p < 0.05

    # setup contingency table
    contingency_table = pd.crosstab(df["A"], df["D"])
    # Chi-square test of independence where variables are dependent
    c, p, dof, expected = chi2_contingency(contingency_table)
    assert p > 0.05
