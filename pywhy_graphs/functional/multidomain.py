from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from pywhy_graphs.algorithms import (
    add_all_snode_combinations,
    compute_invariant_domains_per_node,
    find_connected_pairs,
)
from pywhy_graphs.classes import AugmentedGraph
from pywhy_graphs.functional.utils import _preprocess_parameter_inputs
from pywhy_graphs.typing import Node

from .additive import generate_edge_functions_for_node
from .linear import generate_noise_for_node


def make_random_multidomain_graph(
    G: nx.DiGraph,
    n_domains: int = 2,
    n_nodes_with_s_nodes: Union[int, Tuple[int]] = 1,
    n_invariances_to_try: int = 1,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: Optional[List[Callable[[float], float]]] = None,
    edge_weight_lims: Optional[List[float]] = None,
    random_state=None,
) -> nx.DiGraph:
    r"""Convert an existing linear Gaussian DAG to a multi-domain selection diagram model.

    The multi-domain selection diagram model is a generalization of the regular causal
    diagram in that S-nodes represent possible changes in mechanisms for the underlying node.
    In particular, missing S-node edges to a specific node implies invariances in the
    distribution of that node across domain. For example, if you have a graph
    :math:`X \rightarrow Y`, then the S-node :math:`S^{1,2} \rightarrow Y` represents
    the change in the distribution of :math:`Y` given a change in domain. If there is no
    S-node :math:`S^{1,2} \rightarrow Y`, then the distribution of :math:`Y` is invariant
    across domain 1 and 2.

    Parameters
    ----------
    G : NetworkX DiGraph
        The graph to sample data from. The graph will be modified in-place
        to get the weights and functions of the edges.
    n_domains : int
        The number of domains to split the graph into. By default 2.
    n_nodes_with_s_nodes : int | tuple[int]
        The number of nodes to have S-node edges. By default 1. If a tuple, then will sample
        uniformly a number between the two values.
    n_invariances_to_try : int
        The number of invariances to try to set by deleting S-nodes. By default 1. More S-nodes than
        what is specified by this parameter may be deleted if there are inconsistencies in the
        S-nodes. See Notes for details.
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

    Notes
    -----
    To determine the missing S-node structure, we first construct all possible S-nodes given
    the number of domains, ``n_domains``. The total number of S-nodes will then be
    :math:`\binom{n_{domains}}{2}`. Then, we randomly sample a subset of nodes in the graph
    with S-node edges. The remaining nodes will be missing S-node edges. Then among the nodes
    with S-node edges, we will randomly sample a subset of S-nodes to be missing edges.

    At this stage, there may be inconsistency in the S-nodes connected still. For example,
    if we have the S-nodes :math:`S^{1,2} \rightarrow Y` among 3 domains, then we must
    have either one of the other S-nodes, or none at all. This is because the missing
    :math:`S^{2,3} \rightarrow Y` and :math:`S^{1,3} \rightarrow Y` implies that the
    distribution of :math:`Y` is invariant across domains 1 and 3 and 2 and 3, which
    also implies they are invariant between domain 1 and 3. To fix this, for each node
    with S-node connections, we will delete random set of S-nodes and construct a connected
    component of the S-nodes domains to then remove any remaining S-nodes to keep the
    graph consistent.
    """
    G = G.copy()

    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G
        G = AugmentedGraph(incoming_directed_edges=G)

    if not nx.is_directed_acyclic_graph(directed_G):
        raise ValueError("The input graph must be a DAG.")
    if not G.graph.get("linear_gaussian", True):
        raise ValueError("The input graph must be a linear Gaussian graph.")
    if not isinstance(n_nodes_with_s_nodes, tuple):
        n_nodes_with_s_nodes_ = (n_nodes_with_s_nodes, n_nodes_with_s_nodes)

    rng = np.random.default_rng(random_state)

    s_node_domains = dict()

    # choose nodes with S-nodes
    n_nodes = rng.integers(n_nodes_with_s_nodes_[0], n_nodes_with_s_nodes_[1] + 1)

    # choose the nodes to have S-nodes at random
    node_idx = rng.integers(0, G.number_of_nodes(), size=n_nodes)
    nodes_with_s_nodes = [G.nodes(data=False)[idx] for idx in node_idx]

    # compute all possible S-nodes given the number of domains
    G, s_node_domains = add_all_snode_combinations(G, n_domains)
    all_poss_snodes = set(G.s_nodes)
    for node in nodes_with_s_nodes:
        for s_node in all_poss_snodes:
            # XXX: maybe use directed_G?
            G.add_edge(s_node, node)
    s_nodes = G.s_nodes

    # loop through each node with S-nodes
    for node in nodes_with_s_nodes:
        if n_invariances_to_try == 0 or len(s_nodes) == 0:
            indices = []
            remove_s_node = []
        else:
            indices = rng.integers(len(s_nodes), size=n_invariances_to_try)
            remove_s_node = [s_nodes[idx] for idx in indices]

        # find all connected pairs
        tuples = []
        for s_node in remove_s_node:
            source_domain, target_domain = G.nodes(data=True)[s_node]["domain_ids"]
            tuples.append((source_domain, target_domain))

        connected_pairs = find_connected_pairs(tuples, n_domains)
        invariant_domains = set()
        for domain_pair in connected_pairs:
            # remove all the S-nodes that are not in the connected component
            s_node = s_node_domains[domain_pair]
            G.remove_edge(s_node, node)

            # for each removed S-node, there are invariances in the SCM for this node
            invariant_domains.add(domain_pair[0])
            invariant_domains.add(domain_pair[1])

        # now set the functional relationships based on the invariances
        G.nodes()[node]["invariant_domains"] = invariant_domains

        # now set a random function for each domain that is not invariant
        generate_multidomain_noise_for_node(
            G,
            node,
            n_domains=n_domains,
            node_mean_lims=node_mean_lims,
            node_std_lims=node_std_lims,
            random_state=random_state,
            check_s_node_consistency=False,
        )

        # sample edge functions and weights as a function of the parents
        generate_edge_functions_for_node(
            G,
            node=node,
            edge_weight_lims=edge_weight_lims,
            edge_functions=edge_functions,
            random_state=random_state,
        )

    G.graph["functional"] = "linear_gaussian"
    G.graph["S-nodes"] = s_nodes
    G.graph["n_domains"] = n_domains
    return G


def generate_multidomain_noise_for_node(
    G,
    node: Node,
    n_domains: int,
    node_mean_lims,
    node_std_lims,
    check_s_node_consistency: bool = True,
    random_state=None,
):
    """Sample a linear function for the exogenous noise of a node with S-nodes.

    Parameters
    ----------
    G : AugmentedGraph
        The selection diagram to sample from.
    node : Node
        The node to sample exogenous noise for.
    n_domains : int
        The number of domains to sample from.
    node_mean_lims : Optional[List[float]], optional
        The lower and upper bounds of the mean of the Gaussian random variable, by default None,
        which defaults to a mean of 0.
    node_std_lims : Optional[List[float]], optional
        The lower and upper bounds of the std of the Gaussian random variable, by default None,
        which defaults to a std of 1.
    check_s_node_consistency : bool, optional
        Whether to check that the S-nodes are consistent with the invariant domains, by default
        True.
    random_state : int, optional
        Random seed, by default None.

    Returns
    -------
    _type_
        _description_
    """
    rng = np.random.default_rng(random_state)

    if check_s_node_consistency:
        # compute all possible S-nodes given the number of domains
        G, _ = add_all_snode_combinations(G, n_domains)

        # for each node with S-nodes and compute the invariant domains
        G = compute_invariant_domains_per_node(G, node, n_domains=n_domains)
    else:
        if "invariant_domains" not in G.nodes()[node]:
            raise ValueError("Must specify invariant domains for node {}.".format(node))

    # compute the invariant domains
    invariant_domains = G.nodes()[node]["invariant_domains"]

    # now set a random function for each domain that is not invariant
    domain_noise_params = dict()
    for idx, domain_id in enumerate(range(1, n_domains + 1)):
        if domain_id in invariant_domains:
            continue

        domain_mean_lims = node_mean_lims[idx]
        domain_std_lims = node_std_lims[idx]

        # for domains that are not invariant, we need to set the noise function to a
        # new random function
        mean = rng.uniform(low=domain_mean_lims[0], high=domain_mean_lims[1])
        std = rng.uniform(low=domain_std_lims[0], high=domain_std_lims[1])

        # set the node attribute "functions" to hold the weight and function wrt each parent
        domain_noise_params[domain_id] = {"mean": mean, "std": std}
    G.nodes()[node]["domain_gaussian_noise_function"] = domain_noise_params
    return G


def sample_multidomain_lin_functions(
    G: AugmentedGraph,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: Optional[List[Callable[[float], float]]] = None,
    edge_weight_lims: Optional[List[float]] = None,
    random_state=None,
):
    """Given a selection diagram, sample linear functions for each node.

    Parameters
    ----------
    G : AugmentedGraph
        The selection diagram to sample from. Should have S-nodes to indicate
        different domains.
    node_mean_lims : Optional[List[float]], optional
        The lower and upper bounds of the mean of the Gaussian random variable, by default None,
        which defaults to a mean of 0. If there is a list, then it should be a list of
        length ``n_domains`` meaning that each domain has a different mean range.
    node_std_lims : Optional[List[float]], optional
        The lower and upper bounds of the std of the Gaussian random variable, by default None,
        which defaults to a std of 1. If there is a list, then it should be a list of
        length ``n_domains`` meaning that each domain has a different std range.
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
    G : AugmentedGraph
        The selection diagram with the sampled functions and weights.
    """
    s_node_domains = dict()
    n_domains = len(G.domains)
    s_nodes = set(G.s_nodes)
    if len(s_nodes) == 0:
        return G

    (
        node_mean_lims_,
        node_std_lims_,
        edge_functions_,
        edge_weight_lims_,
    ) = _preprocess_parameter_inputs(
        node_mean_lims,
        node_std_lims,
        edge_functions,
        edge_weight_lims,
        multi_domain=True,
        n_domains=n_domains,
    )

    # compute all nodes that have S-node connections
    nodes_with_s_nodes = []
    for s_node in s_nodes:
        if "domain_ids" not in G.nodes(data=True)[s_node]:
            raise ValueError("Must specify domain_ids for S-node {}.".format(s_node))

        domain_pair = G.nodes(data=True)[s_node]["domain_ids"]
        s_node_domains[domain_pair] = s_node
        nodes_with_s_nodes.extend(
            [node for node in G.successors(s_node) if G.has_edge(s_node, node)]
        )

    # compute all possible S-nodes given the number of domains
    G, s_node_domains = add_all_snode_combinations(G, n_domains)
    for node in G.nodes:
        if node in nodes_with_s_nodes:
            # for each node with S-nodes and compute the invariant domains
            G = compute_invariant_domains_per_node(G, node, n_domains=n_domains)

            # now set a random function for each domain that is not invariant
            G = generate_multidomain_noise_for_node(
                G,
                node,
                n_domains=n_domains,
                node_mean_lims=node_mean_lims_,
                node_std_lims=node_std_lims_,
                random_state=random_state,
                check_s_node_consistency=False,
            )
        else:
            # sample single-domain noise
            G = generate_noise_for_node(
                G, node, node_mean_lims_[0], node_std_lims_[0], random_state=random_state
            )

        # sample edge functions and weights as a function of the parents
        generate_edge_functions_for_node(
            G,
            node=node,
            edge_weight_lims=edge_weight_lims_,
            edge_functions=edge_functions_,
            random_state=random_state,
        )

    G.graph["linear_gaussian"] = True
    G.graph["n_domains"] = n_domains
    return G
