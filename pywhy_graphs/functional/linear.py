from typing import Callable, List, Optional

import networkx as nx
import numpy as np


def make_graph_linear_gaussian(
    G: nx.DiGraph,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: List[Callable[[float], float]] = None,
    edge_weight_lims: Optional[List[float]] = None,
    random_state=None,
):
    r"""Convert an existing DAG to a linear Gaussian graphical model.

    All nodes are sampled from a normal distribution with parametrizations
    defined uniformly at random between the limits set by the input parameters.
    The edges apply then a weight and a function based on the inputs in an additive fashion.
    For node :math:`X_i`, we have:

    .. math::

        X_i = \\sum_{j \in parents} w_j f_j(X_j) + \\epsilon_i

    where:

    - :math:`\\epsilon_i \sim N(\mu_i, \sigma_i)`
    - :math:`w_j \sim U(\\text{edge_weight_lims})`
    - :math:`f_j` is a function sampled uniformly at random
        from `edge_functions`

    Parameters
    ----------
    G : NetworkX DiGraph
        The graph to sample data from. The graph will be modified in-place
        to get the weights and functions of the edges.
    node_mean_lims : Optional[List[float]], optional
        The lower and upper bounds of the mean of the Gaussian random variable, by default None,
        which defaults to a mean of 0.
    node_std_lims : Optional[List[float]], optional
        The lower and upper bounds of the std of the Gaussian random variable, by default None,
        which defaults to a std of 1.
    edge_functions : List[Callable[float]], optional
        The set of edge functions that take in an iid sample from the parent and computes
        a transformation (possibly nonlinear), such as ``(lambda x: x**2, lambda x: x)``,
        by default None, which defaults to the identity function ``lambda x: x``.
    edge_weight_lims : Optional[List[float]], optional
        The lower and upper bounds of the edge weight, by default None,
        which defaults to a weight of 1.
    random_state : int, optional
        Random seed, by default None.

    Returns
    -------
    G : NetworkX DiGraph
        NetworkX graph with the edge weights and functions set with node attributes
        set with ``'parent_functions'``, and ``'gaussian_noise_function'``. Moreover
        the graph attribute ``'linear_gaussian'`` is set to ``True``. One can then
        sample from this graph using :func:`pywhy_graphs.functional.sample_from_graph`.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph must be a DAG.")
    rng = np.random.default_rng(random_state)

    if node_mean_lims is None:
        node_mean_lims = [0, 0]
    elif len(node_mean_lims) != 2:
        raise ValueError("node_mean_lims must be a list of length 2.")
    if node_std_lims is None:
        node_std_lims = [1, 1]
    elif len(node_std_lims) != 2:
        raise ValueError("node_std_lims must be a list of length 2.")
    if edge_functions is None:
        edge_functions = [lambda x: x]
    if edge_weight_lims is None:
        edge_weight_lims = [1, 1]
    elif len(edge_weight_lims) != 2:
        raise ValueError("edge_weight_lims must be a list of length 2.")

    # Create list of topologically sorted nodes
    top_sort_idx = list(nx.topological_sort(G))

    for node_idx in top_sort_idx:
        # get all parents
        parents = sorted(list(G.predecessors(node_idx)))

        # sample noise
        mean = rng.uniform(low=node_mean_lims[0], high=node_mean_lims[1])
        std = rng.uniform(low=node_std_lims[0], high=node_std_lims[1])

        # sample weight and edge function for each parent
        node_function = dict()
        for parent in parents:
            weight = rng.uniform(low=edge_weight_lims[0], high=edge_weight_lims[1])
            func = rng.choice(edge_functions)
            node_function.update({parent: {"weight": weight, "func": func}})

        # set the node attribute "functions" to hold the weight and function wrt each parent
        nx.set_node_attributes(G, {node_idx: node_function}, "parent_functions")
        nx.set_node_attributes(G, {node_idx: {"mean": mean, "std": std}}, "gaussian_noise_function")
    G.graph["linear_gaussian"] = True
    return G
