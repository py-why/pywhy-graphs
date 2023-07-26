from itertools import combinations
from typing import Optional, Set
from warnings import warn

from pywhy_graphs.classes import AugmentedGraph
from pywhy_graphs.typing import Node


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
    s_node_domains = dict()

    sdx = 0
    # add all the S-nodes representing differences across pairs of domains
    # to every single node with S-nodes
    for domains in combinations(range(1, n_domains + 1), 2):
        source_domain, target_domain = sorted(domains)

        # now modify the function of the edge, S-nodes are pointing to
        s_node = ("S", sdx)
        if s_node in G.s_nodes:
            if on_error == "raise":
                raise RuntimeError(f"There is already an S-node {s_node} in G!")
            elif on_error == "warn":
                warn(f"There is already an S-node {s_node} in G!")

        G.add_node(s_node, domain_ids=(source_domain, target_domain))
        s_node_domains[(source_domain, target_domain)] = s_node

        # add S-nodes
        G.graph["S-nodes"][s_node] = (source_domain, target_domain)

        # increment the S-node counter
        sdx += 1
    return G, s_node_domains


# XXX: does not work?
def compute_invariant_domains_per_node(
    G: AugmentedGraph,
    node: Node,
    all_poss_snodes: Optional[Set] = None,
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
    all_poss_snodes : Optional[Set], optional
        All possible S-nodes, by default None. If None, will infer based on the
        number of domains.
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
    # infer the number of domains based on the number of domain IDs in the augmented
    # graph so far
    if n_domains is None:
        n_domains = len(G.domain_ids)

    # original S-nodes
    orig_s_nodes = set(G.s_nodes)

    # add now all relevant S-nodes considering the domains
    if all_poss_snodes is None:
        G_copy, s_node_domains = add_all_snode_combinations(G.copy(), n_domains, on_error="ignore")
        all_poss_snodes = set(G_copy.s_nodes)

    remove_s_node = []
    for s_node in all_poss_snodes:
        if s_node not in orig_s_nodes:
            remove_s_node.append(s_node)

    # find all connected pairs
    tuples = []
    for s_node in remove_s_node:
        source_domain, target_domain = G.nodes(data=True)[s_node]["domain_ids"]
        tuples.append((source_domain, target_domain))
        G.remove_node(s_node)

    # now compute all invariant domains
    connected_pairs = find_connected_pairs(tuples, n_domains)
    invariant_domains = set()
    for domain_pair in connected_pairs:
        # remove all the S-nodes that are not in the connected component
        s_node = s_node_domains[domain_pair]
        G.remove_edge(s_node, node)

        # check if any S-nodes are not in the original
        if s_node not in orig_s_nodes:
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
