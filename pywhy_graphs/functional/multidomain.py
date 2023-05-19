from itertools import combinations
from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


class DisjointSet:
    """Helper data structure to enable disjoint set."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y

    def get_sets(self):
        sets = {}
        for item in self.parent:
            root = self.find(item)
            if root not in sets:
                sets[root] = set()
            sets[root].add(item)
        return sets.values()


def find_connected_pairs(tuples, max_number):
    """Find connected pairs of domain tuples.

    Parameters
    ----------
    tuples : List of tuples
        List of tuples of domain ids (i, j).
    max_number : int
        The maximum number that can be in a domain id.

    Returns
    -------
    connected_pairs : set of tuples
        Set of domain ids that are connected.
    """
    # XXX: this could be made more efficient as it checks for any unordered pair combination
    # in our setting, we always know (j > i) in (i, j).
    disjoint_set = DisjointSet()
    for i, j in tuples:
        if j > max_number:
            continue
        disjoint_set.union(i, j)
    connected_pairs = set()
    for set_items in disjoint_set.get_sets():
        for i in set_items:
            for j in set_items:
                if i != j:
                    connected_pairs.add(tuple(sorted((i, j))))
    return connected_pairs


def make_graph_multidomain(
    G: nx.DiGraph,
    n_domains: int = 2,
    n_nodes_with_s_nodes: Union[int, Tuple[int]] = 1,
    n_invariances_to_try: int = 1,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: List[Callable[[float], float]] = None,
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

    if not isinstance(n_nodes_with_s_nodes, tuple):
        n_nodes_with_s_nodes_ = (n_nodes_with_s_nodes, n_nodes_with_s_nodes)

    s_nodes = []
    s_node_domains = dict()

    # counter on S-nodes
    sdx = 0

    # choose nodes with S-nodes
    n_nodes = rng.integers(n_nodes_with_s_nodes_[0], n_nodes_with_s_nodes_[1] + 1)

    # choose the nodes to have S-nodes at random
    nodes_with_s_nodes = rng.choice(G.nodes, size=n_nodes, replace=False)

    # add all the S-nodes representing differences across pairs of domains
    # to every single node with S-nodes
    for domains in combinations(range(1, n_domains + 1), 2):
        source_domain, target_domain = sorted(domains)

        # now modify the function of the edge, S-nodes are pointing to
        s_node = ("S", sdx)
        G.add_node(s_node, domain_ids=(source_domain, target_domain))
        s_node_domains[(source_domain, target_domain)] = s_node
        for node in nodes_with_s_nodes:
            # the source domain is always the "reference" distribution, that is
            # the one we keep fixed
            G.add_edge(s_node, node)
        # increment the S-node counter
        sdx += 1
        s_nodes.append(s_node)

    # loop through each node with S-nodes
    for node in nodes_with_s_nodes:
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
        nx.set_node_attributes(G, {node: invariant_domains}, "invariant_domains")

        for domain_id in range(1, n_domains + 1):
            if domain_id in invariant_domains:
                continue

            # for domains that are not invariant, we need to set the noise function to a
            # new random function
            mean = rng.uniform(low=node_mean_lims[0], high=node_mean_lims[1])
            std = rng.uniform(low=node_std_lims[0], high=node_std_lims[1])

            # set the node attribute "functions" to hold the weight and function wrt each parent
            nx.set_node_attributes(
                G,
                {node: {target_domain: {"mean": mean, "std": std}}},
                "domain_gaussian_noise_function",
            )

    G.graph["linear_gaussian"] = True
    G.graph["S-nodes"] = s_nodes
    G.graph["n_domains"] = n_domains
    return G
