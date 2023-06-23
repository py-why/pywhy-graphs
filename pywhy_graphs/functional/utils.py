import networkx as nx
import numpy as np


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
