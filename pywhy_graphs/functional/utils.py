import itertools

import networkx as nx
import numpy as np


def to_pgmpy_bayesian_network(G):
    """Convert a discrete graph to a pgmpy Bayesian network.

    Parameters
    ----------
    G : nx.DiGraph
        The discrete graph.

    Returns
    -------
    model : pgmpy.models.BayesianModel
        The Bayesian network.

    Notes
    -----
    This function allows us to use the pgmpy library for inference and sampling.
    For example, one can forward sample from the Bayesian network as follows:

    >>> from pgmpy.sampling import BayesianModelSampling
    >>> inference = BayesianModelSampling(model)
    >>> df = inference.forward_sample(size=5000)
    """
    if G.graph.get("functional") != "discrete":
        raise ValueError("Model is not a functional model.")

    from pgmpy.models import BayesianNetwork

    model = BayesianNetwork(G)
    for node in G.nodes:
        model.add_cpds(G.nodes[node]["cpd"])
    return model


def pre_compute_reduce_maps(G, variable):
    """
    Get probability array-maps for a node as function of conditional dependencies

    Parameters
    ----------
    G : nx.DiGraph
        The discrete graph.
    variable: Bayesian Model Node
        node of the Bayesian network

    Returns
    -------
    state_index : dictionary
        With probability array-index for node as function of conditional dependency values.
    index_to_weight : dictionary
        With mapping of probability array-index to probability array.
    """
    variable_cpd = get_cpd(G, variable)
    variable_evid = variable_cpd.variables[:0:-1]

    # compute all possible state combinations for the parent variables
    state_combinations = [
        tuple(sc)
        for sc in itertools.product(*[range(get_cardinality(G, var)) for var in variable_evid])
    ]

    # compute weights for all possible state combinations
    weights_list = np.array(
        [
            variable_cpd.reduce(
                list(zip(variable_evid, sc)), inplace=False, show_warnings=False
            ).values
            for sc in state_combinations
        ]
    )

    unique_weights, weights_indices = np.unique(weights_list, axis=0, return_inverse=True)

    # convert weights to index; make mapping of state to index
    state_to_index = dict(zip(state_combinations, weights_indices))

    # make mapping of index to weights
    index_to_weight = dict(enumerate(unique_weights))

    # return mappings of state to index, and index to weight
    return state_to_index, index_to_weight


def get_cpd(G, node):
    """Get the CPD associated with a node.

    Parameters
    ----------
    node : Node
        The node for which the CPD is to be returned.

    Returns
    -------
    cpd : TabularCPD
        The CPD associated with the node.
    """
    # Check if the node is present in the graph
    if node not in G:
        raise ValueError(f"Node {node} not present in the graph.")

    # Check if the graph is functional
    if not G.graph.get("functional", False) == "discrete":
        raise ValueError("Model is not a functional model.")

    cpd = G.nodes[node].get("cpd", None)
    if cpd is None:
        raise RuntimeError(f"No CPD associated with {node}.")
    return cpd


def get_cardinality(G, node):
    from pgmpy.factors.discrete import TabularCPD

    cpd: TabularCPD = get_cpd(G, node)
    return cpd.cardinality[0]


def check_discrete_model(G):
    """
    Check the model for various errors. This method checks for the following
    errors.

    * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
    * Checks if the CPDs associated with nodes are consistent with their parents.

    Parameters
    ----------
    G : DiGraph
        The graph to be checked.

    Returns
    -------
    check : boolean
        True if all the checks pass otherwise should throw an error.
    """
    from pgmpy.factors.continuous import ContinuousFactor
    from pgmpy.factors.discrete import TabularCPD

    if not G.graph.get("functional", False) == "discrete":
        raise ValueError("Model is not a functional model.")

    for node in G.nodes:
        cpd = get_cpd(G, node=node)

        # Check if a CPD is associated with every node.
        if cpd is None:
            raise ValueError(f"No CPD associated with {node}")

        # Check if the CPD is an instance of either TabularCPD or ContinuousFactor.
        elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
            evidence = cpd.get_evidence()
            parents = G.predecessors(node)

            # Check if the evidence set of the CPD is same as its parents.
            if set(evidence) != set(parents):
                raise ValueError(
                    f"CPD associated with {node} doesn't have proper parents associated with it."
                )

            # Check if the values of the CPD sum to 1.
            if not cpd.is_valid_cpd():
                raise ValueError(
                    f"Sum or integral of conditional probabilities for node {node} "
                    f"is not equal to 1."
                )

            if len(set(cpd.variables) - set(cpd.state_names.keys())) > 0:
                raise ValueError(
                    f"CPD for {node} doesn't have state names defined for all the variables."
                )

        for index, node in enumerate(cpd.variables[1:]):
            parent_cpd = get_cpd(G, node)
            # Check if the evidence cardinality specified is same as parent's cardinality
            if parent_cpd.cardinality[0] != cpd.cardinality[1 + index]:
                raise ValueError(f"The cardinality of {node} doesn't match in it's child nodes.")
            # Check if the state_names are the same in parent and child CPDs.
            if parent_cpd.state_names[node] != cpd.state_names[node]:
                raise ValueError(f"The state names of {node} doesn't match in it's child nodes.")

    return True


def set_node_attributes_with_G(G1, G2, node):
    """Set node attributes in G1 using G2.

    Parameters
    ----------
    G1 : Graph
        The target graph that is modified.
    G2 : Graph
        The source graph that is used to set the node attributes.
    node : Node
        The specific node to set.
    """
    # get the node attributes of node in G2
    src_node_attrs = G2.nodes(data=True)[node]

    # get the target node attrs
    target_node_attrs = G1.nodes(data=True)[node]

    # update the node attributes with respect
    target_node_attrs.update(src_node_attrs)

    nx.set_node_attributes(G1, {node: target_node_attrs})
    return G1


def _preprocess_parameter_inputs(
    node_mean_lims,
    node_std_lims,
    edge_functions,
    edge_weight_lims,
    multi_domain: bool = False,
    n_domains: int = None,
):
    """Helper function to preprocess common parameter inputs for sampling functional graphs.

    Nodes' exogenous variables are sampled from a Gaussian distribution.
    Edges are sampled, such that an additive linear model is assumed. Note
    the edge functions may be nonlinear, but how they are combined for each
    node as a function of its parents is linear.
    """
    if node_mean_lims is None:
        node_mean_lims = [0, 0]
    if node_std_lims is None:
        node_std_lims = [1, 1]
    if edge_functions is None:
        edge_functions = [lambda x: x]
    if edge_weight_lims is None:
        edge_weight_lims = [1, 1]

    if not multi_domain:
        for param in [node_mean_lims, node_std_lims, edge_weight_lims]:
            if len(param) != 2:
                raise ValueError(f"{param} must be a list of length 2.")
    elif multi_domain:
        for param in [node_mean_lims, node_std_lims, edge_weight_lims]:
            if len(param) != n_domains:
                raise ValueError(f"{param} must be a list of length 2 or {n_domains} domains.")

        # if the parameters are not a list of length n_domains, then they must be a
        # list of length n_domains
        if len(node_mean_lims) != n_domains:
            node_mean_lims = [node_mean_lims] * n_domains  # type: ignore
        if len(node_std_lims) != n_domains:
            node_std_lims = [node_std_lims] * n_domains  # type: ignore
        if len(edge_weight_lims) != n_domains:
            edge_weight_lims = [edge_weight_lims] * n_domains  # type: ignore
    return node_mean_lims, node_std_lims, edge_functions, edge_weight_lims


def _preprocess_md_parameter_inputs(
    node_mean_lims, node_std_lims, edge_functions, edge_weight_lims, n_domains: int
):
    """Helper function to preprocess common parameter inputs for sampling functional graphs.

    Nodes' exogenous variables are sampled from a Gaussian distribution.
    Edges are sampled, such that an additive linear model is assumed. Note
    the edge functions may be nonlinear, but how they are combined for each
    node as a function of its parents is linear.

    Parameters are encoded as a 2D array with rows as the domains and columns as the lower and
    upper bound.

    Parameters
    ----------
    node_mean_lims : ArrayLike of shape (n_domains, 2)
        A 2D array with rows as the domains and columns as the lower and upper bound
        of the node's exogenous variable mean. If None, then initialized to [0, 1]
        for every domain.
    node_std_lims : ArrayLike of shape (n_domains, 2)
        A 2D array with rows as the domains and columns as the lower and upper initialized
        of the node's exogenous variable standard deviation. If None, then initialized
        to [0.1, 1.0] for every domain.
    edge_functions : ArrayLike of shape (n_functions,)
        A set of different lambda functions.
    edge_weight_lims : ArrayLike of shape (n_domains, 2)
        A 2D array with rows as the domains and columns as the lower and upper bound
        of the edge weights that are used to combine the parents' values linearly. If None,
        then initialized to [-1, 1] for every domain.
    n_domains : int
        Number of domains.
    """
    # initialize to default values
    if node_mean_lims is None:
        node_mean_lims = np.zeros((n_domains, 2))
        node_mean_lims[:, 1] = 1
    if node_std_lims is None:
        node_std_lims = np.zeros((n_domains, 2))
        node_std_lims[:, 1] = 1
        node_std_lims[:, 0] = 0.1
    if edge_functions is None:
        edge_functions = [lambda x: x]
    if edge_weight_lims is None:
        edge_weight_lims = np.zeros((n_domains, 2))
        edge_weight_lims[:, 0] = -1
        edge_weight_lims[:, 1] = 1

    for param in [node_mean_lims, node_std_lims, edge_weight_lims]:
        if len(param) != n_domains:
            raise ValueError(f"{param} must be a list of length 2 or {n_domains} domains.")
    return node_mean_lims, node_std_lims, edge_functions, edge_weight_lims
