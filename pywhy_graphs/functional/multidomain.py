import collections
from itertools import combinations
from typing import Callable, List, Optional, Set

import networkx as nx
import numpy as np


def make_graph_multidomain(
    G: nx.DiGraph,
    n_domains: int = 2,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: List[Callable[[float], float]] = None,
    edge_weight_lims: Optional[List[float]] = None,
    random_state=None,
) -> nx.DiGraph:
    r"""Convert an existing linear Gaussian DAG to a multi-domain selection diagram model.

    # XXX: This is a work in progress. The idea is to split the graph into n_domains
    # represented by S-nodes. However, it is difficult to do this in an efficient general
    # way that works given some random assignment of S-node (source, target) domains.
    # For example, if you have 5 domains, and all possible (source, target) pairs of corresponding
    # S-nodes are added, except (2, 3), and (3, 4) to X.
    # X -> Y, with then (5 choose 2) - 2 S-nodes pointing to X.
    # This would imply domain 2, 3 and 4 are the same for X's distribution.
    # However, then the S-node correpsonding to (2, 4) should also not be present, since that
    # would imply a difference between domain 2 and 4 for X's distribution, even though they are
    # the same as domain 3.

    Parameters
    ----------
    G : NetworkX DiGraph
        The graph to sample data from. The graph will be modified in-place
        to get the weights and functions of the edges.
    n_domains : int
        The number of domains to split the graph into.
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

    See Also
    --------
    make_graph_linear_gaussian : Create a linear Gaussian graph
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph must be a DAG.")
    if not G.graph.get("linear_gaussian", True):
        raise ValueError("The input graph must be a linear Gaussian graph.")

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

    s_nodes = []
    s_node_domains = collections.defaultdict(list)

    # counter on S-nodes
    sdx = 0

    # first, add all the S-nodes representing differences across pairs of domains
    for domains in combinations(range(1, n_domains + 1), 2):
        source_domain, target_domain = sorted(domains)

        # choose a random number of S-nodes to add between (source, target)
        n_s_nodes = rng.integers(0, 3)
        s_nodes_pointer = rng.choice(G.nodes, size=n_s_nodes, replace=False)

        # now modify the function of the edge, S-nodes are pointing to
        s_node = ("S", sdx)
        G.add_node(s_node, domain_ids=(source_domain, target_domain))
        for node in s_nodes_pointer:
            # the source domain is always the "reference" distribution, that is
            # the one we keep fixed
            G.add_edge(s_node, node)

            # mape each source to its target and corresponding S-nodes
            s_node_domains[source_domain].append((target_domain, node, s_node))

        # increment the S-node counter
        sdx += 1

        s_nodes.append(s_node)

    # loop through each domain's S-nodes
    for domain in range(1, n_domains + 1):
        for target_domain, node, s_node in s_node_domains[domain]:
            # now modify the function of 'node'
            # sample noise
            current_mean = G.nodes[node]["gaussian_noise_function"]["mean"]
            current_std = G.nodes[node]["gaussian_noise_function"]["std"]

            mean = rng.uniform(low=node_mean_lims[0], high=node_mean_lims[1])
            std = rng.uniform(low=node_std_lims[0], high=node_std_lims[1])

            new_mean = current_mean + mean
            new_std = current_std + std

            # set the node attribute "functions" to hold the weight and function wrt each parent
            nx.set_node_attributes(G, {node: {"mean": mean, "std": std}}, "gaussian_noise_function")
            nx.set_node_attributes(
                G,
                {node: {target_domain: {"mean": new_mean, "std": new_std}}},
                "domain_gaussian_noise_function",
            )

    G.graph["linear_gaussian"] = True
    G.graph["S-nodes"] = s_nodes
    G.graph["n_domains"] = n_domains
    return G
