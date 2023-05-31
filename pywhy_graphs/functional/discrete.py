from typing import Dict, List
import networkx as nx
from pgmpy.factors.discrete import TabularCPD

from pywhy_graphs.typing import Node


def generate_noise_for_node(
    G, node, node_mean_lims, node_std_lims, random_state=None
):
    pass


def add_cpd_for_node(G: nx.DiGraph, node, cpd: TabularCPD, overwrite: bool=False):
    """Add CPD (Conditional Probability Distribution) to graph.

    This is a wrapper around a similar function as BayesianNetwork.add_cpds.
    Adds a conditional probability distribution table for each node, which
    is defines conditional probabilities for that node given its parents'
    states.

    Parameters
    ----------
    G : Graph
        The causal graph.
    node : Node
        A node in G.
    cpd  :  TabularCPD
        List of CPDs which will be associated with this node.
    """
    if not isinstance(cpd, (TabularCPD,)):
        raise ValueError("Only pgmpy.TabularCPD can be added.")

    if set(cpd.scope()) - set([node]):
        raise ValueError(
            f"CPD should be defined for {node}. It is not: {cpd}")

    # check if a CPD already exists for node
    if G.nodes[node].get('CPD') is not None and not overwrite:
        raise RuntimeError(f'A CPD exists in G for {node}. Set overwrite to True if you want to overwrite.')

    # check that CPD has evidence using the parents of node
    if set(cpd.get_evidence()) != set(G.predecessors(node)):
        raise ValueError(
            f"CPD should be defined for all parents of {node}: "
            f"{G.predecessors(node)}. It is not: {cpd}")
    
    # check that the CPD has cardinality of the evidence that matches the cardinality set
    for cardinality, parent in zip(cpd.cardinality[1:], cpd.get_evidence()):
        if G.nodes[parent]['cardinality'] != cardinality:
            raise RuntimeError(f'The cardinality of parent variable {parent} - {G.nodes[parent]["cardinality"]} '
                               f'does not match the cardinality of the passed in CPT {cardinality}')

    # assign the conditional probability distribution
    G.nodes[node]['cpd'] = cpd
    G.nodes[node]['cardinality'] = cpd.cardinality[0]
    G.graph['functional'] = 'discrete'
    return G


def make_random_discrete_graph(G,
                                cardinality_lims: List[int]=None,
                                weight_lims: List[int] = None,
                                noise_ratio_lims: List[float] = None,
    random_state=None,
) -> nx.DiGraph:
    if cardinalities is not None and not all(node in cardinalities for node in G.nodes):
        raise RuntimeError(f'Cardinalities must be specified for all nodes.')
    else:
        # default is a binary bayesian network
        cardinalities = {node: 2 for node in G.nodes}
