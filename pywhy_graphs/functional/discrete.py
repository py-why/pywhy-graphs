from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD

from pywhy_graphs.typing import Node


def apply_discrete_soft_intervention(
    G, targets: Set[Node], weight_ranges: Optional[List] = None, random_state=None
):
    """Apply a soft intervention that changes the underlying CPD distribution of the target nodes.

    Parameters
    ----------
    G : Graph
        Linear functional causal graph.
    targets : Set[Node]
        The set of nodes to intervene on simultanenously.
    random_state : RandomState, optional
        Random seed, by default None.

    Returns
    -------
    G : Graph
        The functional linear causal graph with the intervention applied on the
        target nodes. The perturbation occurs by altering the conditional probability distribution
        table of the targets. We resample an entirely new CPD for each target node.
    """
    if not G.graph.get("functional") == "discrete":
        raise ValueError("The input graph must be a discrete graph.")
    if not all(target in G.nodes for target in targets):
        raise ValueError(f"All targets {targets} must be in the graph: {G.nodes}.")
    if weight_ranges is not None and len(weight_ranges) != len(targets):
        raise ValueError(
            f"weight_ranges must be None or have the same length as targets: {targets}."
        )

    rng = np.random.default_rng(random_state)

    for idx, target in enumerate(targets):
        cpd: TabularCPD = G.nodes[target]["cpd"]

        # maintain the same cardinality for the node as previously
        cardinality = cpd.cardinality[0]

        # resample the weight range
        if weight_ranges is None:
            weight_range = [1, 5]
        else:
            weight_range = weight_ranges[idx]

        # resample the CPD
        cpd = _sample_random_cpd(
            G, target, cardinality=cardinality, weight_range=weight_range, rng=rng
        )

        add_cpd_for_node(G, target, cpd, overwrite=True)
    return G


def add_cpd_for_node(
    G: nx.DiGraph,
    node,
    cpd: TabularCPD,
    noise_ratio: float = 0.0,
    random_state=None,
    overwrite: bool = False,
):
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
        CPDs which will be associated with this node.
    noise_ratio : float
        The ratio of the times the noise function is applied to sample the node.
        By default, the exogenous distribution is defined as a uniform distribution over
        all possible values of the node. If noise_ratio is set to 0.1, then 10% of the
        time the exogenous distribution is applied, and 90% of the time the parent
        function is applied.
    random_state : random number generator, optional
        The random number generator, by default None.
    overwrite : bool, optional
        Whether to overwrite an existing CPD for the node, by default False.
    """
    rng = np.random.default_rng(random_state)

    if not isinstance(cpd, (TabularCPD,)):
        raise ValueError("Only pgmpy.TabularCPD can be added.")

    if set([cpd.variable]) - set([node]):
        raise ValueError(f"CPD should be defined for {node}. It is not: {cpd.variable}")

    # check if a CPD already exists for node
    if G.nodes[node].get("cpd") is not None and not overwrite:
        raise RuntimeError(
            f"A CPD exists in G for {node}. Set overwrite to True if you want to overwrite."
        )

    # check that CPD has evidence using the parents of node
    if set(cpd.get_evidence()) != set(G.predecessors(node)):
        raise ValueError(
            f"CPD should be defined for all parents of {node}: "
            f"{G.predecessors(node)}. It is not: {cpd}"
        )

    # parents of the node should have CPD defined first
    for parent in G.predecessors(node):
        if G.nodes[parent].get("cpd") is None:
            raise RuntimeError(f"CPD for parent {parent} of node {node} must be defined first.")

    # check that the CPD has cardinality of the evidence that matches the cardinality set
    for cardinality, parent in zip(cpd.cardinality[1:], cpd.variables[1:]):
        if G.nodes[parent].get("cardinality", 0) != cardinality:
            raise RuntimeError(
                f"The cardinality of parent variable {parent} -"
                f'{G.nodes[parent].get("cardinality")} '
                f"does not match the cardinality of the passed in CPT {cardinality}"
            )

    # possible values taken on by the `node`
    possible_values = cpd.state_names[node]

    # assign the conditional probability distribution
    G.nodes[node]["cpd"] = cpd
    G.nodes[node]["cardinality"] = cpd.cardinality[0]
    G.nodes[node]["possible_values"] = possible_values
    G.graph["functional"] = "discrete"

    # now assign the relevant functions
    if G.nodes[node].get("noise_ratio") is None:
        G.nodes[node]["noise_ratio"] = 0.0

    if G.nodes[node].get("exogenous_function") is None:
        # default is the uniform choice function
        G.nodes[node]["exogenous_function"] = lambda x: x

    if G.nodes[node].get("exogenous_distribution") is None:
        G.nodes[node]["exogenous_distribution"] = lambda: rng.choice(
            a=possible_values, size=None, p=None
        )

    # define the parent function
    parents = np.array(sorted(G.predecessors(node)))

    def parent_func(*args):
        # list of tuples indicating the (variable, variable_state)
        reducing_vals = []
        for idx, parent_val in enumerate(sorted(args)):
            reducing_vals.append((parents[idx], parent_val))

        # now reduce the cpd
        cpd_reduced = cpd.reduce(reducing_vals, inplace=False)
        p = cpd_reduced.get_values().squeeze()
        if not p.size == len(possible_values):
            raise RuntimeError(
                f"The sampled CPT should have length {len(possible_values)},"
                f"but has size {p.size}"
            )
        return rng.choice(a=possible_values, size=None, p=p)

    # set the parent function CPT, or if the node has no parents, set the exogenous distribution
    # as the input CPT
    if len(parents) > 0:
        G.nodes[node]["parent_function"] = parent_func
        G.nodes[node]["noise_ratio"] = noise_ratio
    else:
        G.nodes[node]["exogenous_distribution"] = parent_func
        G.nodes[node]["noise_ratio"] = 1.0

    return G


def make_random_discrete_graph(
    G: nx.DiGraph,
    cardinality_lims: Optional[Dict[Any, List[int]]] = None,
    weight_lims: Optional[Dict[Any, List[int]]] = None,
    noise_ratio_lims: Optional[List[float]] = None,
    overwrite: bool = False,
    random_state=None,
) -> nx.DiGraph:
    """Sample a random discrete graph given a graph structure.

    This function samples a random discrete graph given a graph structure. The
    graph structure is defined by the input graph ``G``. The graph ``G`` must be
    a DAG. The function samples a random cardinality for each node, and then
    samples a random `pgmpy.TabularCPD` for each node given the cardinality of
    the node and the cardinality of the parents of the node. The function then
    assigns the sampled `pgmpy.TabularCPD` to the node, and assigns the
    relevant functions to the node. The relevant functions are the
    ``parent_function``, ``exogenous_function``, and ``exogenous_distribution``.

    To sample the relevant conditional probability values within the CPT, the
    weights are sampled for each discrete category for each variable. The
    weights are sampled and used as the ``p`` input to `numpy.random.choice`,
    which normalizes the weights so that the probabilities they induce are valid.

    The noise ratio is also sampled for each node, and is used to define the
    ``noise_ratio`` attribute of the node.

    Parameters
    ----------
    G : nx.DiGraph
        A defined DAG.
    cardinality_lims : List[int], optional
        The range of cardinalities for the variables, by default None,
        which defaults to binary.
    weight_lims : List[int], optional
        The possible weights to sample each discrete category for each variable,
        by default None, which defaults to a uniform weight. The weights
        are sampled and used as the ``high`` input to `numpy.random.integers`, which
        normalizes the weights so that the probabilities they induce are valid and
        sum to 1.0. The ``low`` is always equal to 0.
    noise_ratio_lims : List[float], optional
        The possible range for the noise ratio, by default None, which will
        default to all variables having noise ratio of 0.0.
    overwrite : bool, optional
        Whether to overwrite the existing attributes of the graph, by default False.
    random_state : RandomGenerator, optional
        The random number generator, by default None.

    Returns
    -------
    G : nx.DiGraph
        The altered functional DAG.
    """
    if cardinality_lims is None:
        # Default to binary variables
        cardinality_lims_ = {node: [1, 2] for node in G.nodes}
    else:
        cardinality_lims_ = cardinality_lims
    if weight_lims is None:
        # Default weight range [1, 2]
        weight_lims_ = {node: [1, 2] for node in G.nodes}
    else:
        weight_lims_ = weight_lims

    rng = np.random.default_rng(random_state)
    if noise_ratio_lims is None:
        # Default noise ratio of 0.0
        noise_ratio_lims_ = {node: rng.uniform(*[0.0, 0.0]) for node in G.nodes}
    else:
        noise_ratio_lims_ = {node: rng.uniform(*noise_ratio_lims[node]) for node in G.nodes}

    if not all([node in cardinality_lims_.keys() for node in G.nodes]):
        raise ValueError("The cardinality limits must be defined for all nodes in the graph")
    if not all([node in weight_lims_.keys() for node in G.nodes]):
        raise ValueError("The weight limits must be defined for all nodes in the graph")
    if not all([node in noise_ratio_lims_.keys() for node in G.nodes]):
        raise ValueError("The noise ratio limits must be defined for all nodes in the graph")

    G = _sample_generative_graph_model(
        G,
        cardinality_lims_,
        weight_lims_,
        noise_ratio_lims_,
        overwrite=overwrite,
        random_state=random_state,
    )

    return G


def _sample_generative_graph_model(
    G, cardinality_lims, weight_lims, noise_ratios, overwrite=False, random_state=None
):
    rng = np.random.default_rng(random_state)

    # sample random CPD for each node in topological order
    for node in nx.topological_sort(G):
        # sample random CPD defined by the cardinality of the node and the cardinality of
        # the parents
        cardinality = rng.integers(low=cardinality_lims[node][0], high=cardinality_lims[node][1])
        cpd = _sample_random_cpd(G, node, cardinality, weight_lims[node], rng)

        # add the CPD to the node attributes
        G = add_cpd_for_node(
            G,
            node,
            cpd,
            noise_ratio=noise_ratios[node],
            random_state=random_state,
            overwrite=overwrite,
        )
    return G


def _sample_random_cpd(G, node, cardinality, weight_range, rng):
    parents = list(G.predecessors(node))

    # now sample a weight per discrete category for each variable given a combination of
    # all parent values
    n_cols = np.prod([G.nodes[parent]["cardinality"] for parent in parents]).astype(int)
    cpd_values = np.zeros((cardinality, n_cols))

    # XXX: We can improve the discriminability of the CPD by sampling the weights
    # unevenly for each category. This will make the CPD more discriminable if the
    # conditional distributions for each category are more different from each other.
    for col_idx in range(n_cols):
        # sample each category with a different upper-bound weight
        weights = []
        prev_weight = weight_range[0]
        for idx in range(cardinality):
            prev_weight = rng.uniform(low=prev_weight, high=weight_range[1])
            weights.append(prev_weight)

        # 1D array of shape (cardinality,)
        p = np.array(weights) / np.sum(weights)
        cpd_values[:, col_idx] = p

    evidence_card = [G.nodes[parent]["cardinality"] for parent in parents]
    cpd = TabularCPD(
        variable=node,
        variable_card=cardinality,
        values=cpd_values,
        evidence=parents,
        evidence_card=evidence_card,
    )
    return cpd
