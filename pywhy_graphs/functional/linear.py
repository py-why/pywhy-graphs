from typing import Callable, List, Set

import networkx as nx
import numpy as np

from pywhy_graphs.typing import Node

from .additive import generate_edge_functions_for_node
from .utils import _preprocess_parameter_inputs


def make_graph_linear_gaussian(
    G: nx.DiGraph,
    node_mean_lims: List[float] = None,
    node_std_lims: List[float] = None,
    edge_functions: List[Callable[[float], float]] = None,
    edge_weight_lims: List[float] = None,
    random_state=None,
) -> nx.DiGraph:
    r"""Convert an existing DAG to a linear Gaussian graphical model.

    All nodes are sampled from a normal distribution with parametrizations
    defined uniformly at random between the limits set by the input parameters.
    The edges apply then a weight and a function based on the inputs in an additive fashion.
    For node :math:`X_i`, we have:

    .. math::

        X_i = \\sum_{j \in parents} w_j f_j(X_j) + \\epsilon_i

    where:

    - :math:`\\epsilon_i \sim N(\mu_i, \sigma_i)`, where :math:`\mu_i` is sampled
        uniformly at random from `node_mean_lims` and :math:`\sigma_i` is sampled
        uniformly at random from `node_std_lims`.
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
        the graph attribute ``'linear_gaussian'`` is set to ``True``.
    """
    G = G.copy()

    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G

    if not nx.is_directed_acyclic_graph(directed_G):
        raise ValueError("The input graph must be a DAG.")
    rng = np.random.default_rng(random_state)

    # preprocess hyperparameters and check for validity
    (
        node_mean_lims_,
        node_std_lims_,
        edge_functions_,
        edge_weight_lims_,
    ) = _preprocess_parameter_inputs(
        node_mean_lims, node_std_lims, edge_functions, edge_weight_lims
    )

    # Create list of topologically sorted nodes
    top_sort_idx = list(nx.topological_sort(directed_G))

    # sample noise and edge functions for each node and its parents
    for node in top_sort_idx:
        # sample noise
        G = generate_noise_for_node(
                G, node, node_mean_lims_, node_std_lims_, random_state=random_state
        )
        
        # sample edge functions and weights
        generate_edge_functions_for_node(
            G,
            node=node,
            edge_weight_lims=edge_weight_lims_,
            edge_functions=edge_functions_,
            random_state=random_state,
        )
    G.graph["linear_gaussian"] = True
    return G

def generate_noise_for_node(
        G, node, node_mean_lims, node_std_lims, random_state=None
):
    rng = np.random.default_rng(random_state)
    
    # sample noise
    mean = rng.uniform(low=node_mean_lims[0], high=node_mean_lims[1])
    std = rng.uniform(low=node_std_lims[0], high=node_std_lims[1])
    G.nodes[node]["gaussian_noise_function"] = {"mean": mean, "std": std}
    return G


def apply_linear_soft_intervention(
    G, targets: Set[Node], type: str = "additive", random_state=None
):
    """Applies a soft intervention to a linear Gaussian graph.

    Parameters
    ----------
    G : Graph
        Linear functional causal graph.
    targets : Set[Node]
        The set of nodes to intervene on simultanenously.
    type : str, optional
        Type of intervention, by default "additive".
    random_state : RandomState, optional
        Random seed, by default None.

    Returns
    -------
    G : Graph
        The functional linear causal graph with the intervention applied on the
        target nodes. The perturbation occurs on the ``gaussian_noise_function``
        of the target nodes. That is, the soft intervention, perturbs the
        exogenous noise of the target nodes.
    """
    if not G.graph.get("linear_gaussian", True):
        raise ValueError("The input graph must be a linear Gaussian graph.")
    if not all(target in G.nodes for target in targets):
        raise ValueError(f"All targets {targets} must be in the graph: {G.nodes}.")

    rng = np.random.default_rng(random_state)

    for target in targets:
        if type == "additive":
            G.nodes[target]["gaussian_noise_function"]["mean"] += rng.uniform(low=-1, high=1)

    return G
