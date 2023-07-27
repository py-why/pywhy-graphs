from itertools import combinations
from typing import Optional
from warnings import warn

from pywhy_graphs.classes import AugmentedGraph
from pywhy_graphs.typing import Node


def get_all_snode_combinations(n_domains):
    """Compute a mapping of domain pairs to all possible S-nodes.

    S-nodes are defined as ``('S', <idx>)``, where ``<idx>`` is an integer
    starting from 0. Each S-node is by construction mapped to a pair of domain
    IDs.

    Parameters
    ----------
    n_domains : int
        The number of possible domains.

    Returns
    -------
    s_node_domains : dict
        A mapping of domain pairs to S-nodes.
    """
    s_node_domains = dict()

    sdx = 0
    # add all the S-nodes representing differences across pairs of domains
    # to every single node with S-nodes
    for domains in combinations(range(1, n_domains + 1), 2):
        source_domain, target_domain = sorted(domains)

        # now modify the function of the edge, S-nodes are pointing to
        s_node = ("S", sdx)
        s_node_domains[(source_domain, target_domain)] = s_node

        # increment the S-node counter
        sdx += 1
    return s_node_domains


def add_all_snode_combinations(G, n_domains: int, on_error="raise"):
    """Add all possible S-nodes to the graph given number of domains.

    Parameters
    ----------
    G : AugmentedGraph
        The augmented graph.
    n_domains : int
        The number of domains.
    on_error : str, optional
        How to handle errors, by default 'raise'. Can be one of:
        - 'raise': raise an exception
        - 'ignore': ignore the error.
        - 'warn': raise a warning

    Returns
    -------
    G : AugmentedGraph
        The augmented graph with all possible S-nodes added. Note that none
        of the added S-nodes have any edges.
    """
    G = G.copy()

    # compute the relevant S-node combinations
    s_node_domains = get_all_snode_combinations(n_domains)

    # add all the S-nodes representing differences across pairs of domains
    # to every single node with S-nodes
    for (source_domain, target_domain), s_node in s_node_domains.items():

        # now modify the function of the edge, S-nodes are pointing to
        s_node_domains[(source_domain, target_domain)] = s_node
        if s_node in G.s_nodes:
            if on_error == "raise":
                raise RuntimeError(f"There is already an S-node {s_node} in G!")
            elif on_error == "warn":
                warn(f"There is already an S-node {s_node} in G!")

        G.add_node(s_node, domain_ids=(source_domain, target_domain))

        # add S-nodes
        G.graph["S-nodes"][s_node] = (source_domain, target_domain)

    return G, s_node_domains


def get_connected_snodes(G, node):
    """Get all the connected S-nodes to a node.

    Parameters
    ----------
    G : AugmentedGraph
        The augmented graph.
    node : Node
        The node to get the connected S-nodes for.

    Returns
    -------
    connected_snodes : Set[Node]
        Set of connected S-nodes.
    """
    connected_snodes = set()
    for s_node in G.s_nodes:
        if G.has_edge(s_node, node):
            connected_snodes.add(s_node)
    return connected_snodes


def remove_snode_edge(G, snode, node, preserve_invariance=True):
    """Remove an S-node edge from a selection diagram.

    The removal of an S-node edge implies invariances in the diagram
    across different domains represented by the S-node. This invariance
    may lead to other invariances in selection diagrams representing
    more than 3 domain.

    Parameters
    ----------
    G : AugmentedGraph
        The augmented graph with S-nodes.
    snode : Node
        A S-node representing a possible difference across two domains.
    node : Node
        The to node of the S-node.
    preserve_invariance : bool, optional
        Whether or not to remove additional S-node edges that are required
        to preserve the relative invariances, by default True.

    Returns
    -------
    G : AugmentedGraph
        Augmented graph with removed S-node edges.
    """
    domain_ids = G.domain_ids
    snode_domains = get_all_snode_combinations(len(domain_ids))

    if snode not in snode_domains.values():
        raise RuntimeError(f"S-node {snode} is not a valid S-node!")

    # remove the edge
    G.remove_edge(snode, node)

    # now compute the connected pairs of domains
    if preserve_invariance:
        # get all the other S-nodes not linked to node
        other_snodes = set()
        domain_pairs = []

        # get all S-nodes with an edge to `node`
        connected_snodes = get_connected_snodes(G, node)

        # find connected pairs of domain IDs that must be invariant
        for domain_pair, snode_ in snode_domains.items():
            if snode_ not in connected_snodes or snode_ == snode:
                other_snodes.add(snode_)
                domain_pairs.append(domain_pair)
        connected_domain_pairs = find_connected_pairs(domain_pairs, len(domain_ids))

        # now remove all the S-node edges for S-nodes that are in the
        # connected component
        for domain_pair in connected_domain_pairs:
            snode_ = snode_domains[domain_pair]
            G.remove_edge(snode_, node)
    return G


def compute_invariant_domains_per_node(
    G: AugmentedGraph,
    node: Node,
    n_domains: Optional[int] = None,
    inconsistency="raise",
):
    """Compute the invariant domains for a specific node.

    This proceeds by constructing all possible S-nodes given the number of domains
    (i.e. ``n_domains choose 2`` S-nodes), and then uses the S-nodes in G to infer
    the invariant domains for the node.

    Parameters
    ----------
    G : AugmentedGraph
        The augmented graph.
    node : Node
        The node in G to compute the invariant domains for.
    n_domains : int, optional
        The number of domains, by default None. If None, will infer based on the
        ``domain_ids`` attribute of G.
    inconsistency : str, optional
        How to handle inconsistencies, by default 'raise'. Can be one of:
        - 'raise': raise an exception
        - 'ignore': ignore the inconsistency.
        - 'warn': raise a warning

        An inconsistency is when the current included S-nodes are not the same
        after computing the invariant domains. If 'ignore', or 'warn', the
        inconsistent S-node will be removed in `G`.

    Returns
    -------
    G : AugmentedGraph
        The augmented graph
    """
    G = G.copy()

    # infer the number of domains based on the number of domain IDs in the augmented
    # graph so far
    if n_domains is None:
        n_domains = len(G.domain_ids)

    # add now all relevant S-nodes considering the domains
    s_node_domains = get_all_snode_combinations(n_domains)

    # get all S-nodes with an edge to `node`
    connected_snodes = get_connected_snodes(G, node)

    # get all the other S-nodes not linked to node
    other_snodes = set()
    domain_pairs = []

    # find connected pairs of domain IDs that must be invariant
    for domain_pair, snode_ in s_node_domains.items():
        if snode_ not in connected_snodes:
            other_snodes.add(snode_)
            domain_pairs.append(domain_pair)
    connected_domain_pairs = find_connected_pairs(domain_pairs, n_domains)

    # now compute all invariant domains
    invariant_domains = set()
    for domain_pair in connected_domain_pairs:
        # remove all the S-nodes that are not in the connected component
        s_node = s_node_domains[domain_pair]

        # if there is an S-node edge that is inconsistent with the invariant
        # domains, then raise an error
        if G.has_edge(s_node, node):
            if inconsistency == "raise":
                raise RuntimeError(f"Inconsistency in S-nodes for node {node}!")
            elif inconsistency == "warn":
                warn(f"Inconsistency in S-nodes for node {node}!")

        # for each removed S-node, there are invariances in the SCM for this node
        invariant_domains.add(domain_pair[0])
        invariant_domains.add(domain_pair[1])

    # now set the functional relationships based on the invariances
    G.nodes()[node]["invariant_domains"] = invariant_domains
    return G


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

    This is useful for removing S-nodes among a selection diagram that represents
    more than 3 domains. For example, if we have 4 domains, we can have the following
    S-nodes: (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4). However, if we removed
    the S-node (1, 2), and (1, 3), we should also remove (2, 3) as it is connected.

    This is because the removal of (1, 2) and (1, 3) implies that domains 1 and 2
    are invariant for some node, and domains 1 and 3 are invariant for that node.
    However, this also implies that domains 2 and 3 are invariant for that node
    by transitivity.

    However, this is only required if we want the S-node to be "strict" as in
    the invariance is definitely not valid when the S-node edge is not present.
    If the S-node edge only implies that the invariance "could" not hold, then
    we do not need to remove the additional S-nodes.

    Parameters
    ----------
    tuples : List of tuples
        List of tuples of domain ids (i, j).
    max_number : int
        The maximum number that can be in a domain id.
        Assumes indexing starts at 1.

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
