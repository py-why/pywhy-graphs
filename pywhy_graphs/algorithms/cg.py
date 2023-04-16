import copy
from collections import OrderedDict, deque

from pywhy_graphs import CG

__all__ = ["is_valid_cg"]


def is_valid_cg(graph: CG):
    """
    Checks if a supplied chain graph is valid.

    This implements the original defintion of a (Lauritzen Wermuth Frydenberg) chain graph as
    presented in [1]_.

    Define a cycle as a series of nodes X_1 -o X_2 ... X_n -o X_1 where the edges may be directed or
    undirected. Note that directed edges in a cycle must all be aligned in the same direction. A
    chain graph may only contain cycles consisting of only undirected edges. Equivalently, a chain
    graph does not contain any cycles with one or more directed edges.

    Parameters
    __________
    graph : CG
        The graph.

    Returns
    _______
    is_valid : bool
        Whether supplied `graph` is a valid chain graph.

    References
    ----------
    .. [1] Frydenberg, Morten. “The Chain Graph Markov Property.” Scandinavian Journal of
    Statistics, vol. 17, no. 4, 1990, pp. 333–53. JSTOR, http://www.jstor.org/stable/4616181.
    Accessed 15 Apr. 2023.


    """

    # Check if directed edges are acyclic
    undirected_edge_name = graph.undirected_edge_name
    directed_edge_name = graph.directed_edge_name
    all_nodes = graph.nodes()
    G_undirected = graph.get_graphs(edge_type=undirected_edge_name)
    G_directed = graph.get_graphs(edge_type=directed_edge_name)

    # Search over all nodes.
    for v in all_nodes:
        queue = deque([])
        # Fill queue with paths from v starting with outgoing directed edge
        # OrderedDict used for O(1) set membership and ordering
        for _, z in G_directed.out_edges(nbunch=v):
            d = OrderedDict()
            d[v] = None
            d[z] = None
            queue.append(d)

        while queue:
            # For each path in queue, progress along edges in certain
            # manner
            path = queue.popleft()
            rev_path = reversed(path)
            last_added = next(rev_path)
            second_last_added = next(rev_path)

            # For directed edges progress is allowed for outgoing edges
            # only
            for _, node in G_directed.out_edges(nbunch=last_added):
                if node in path:
                    return False
                new_path = copy.deepcopy(path)
                new_path[node] = None
                queue.append(new_path)

            # For undirected edges, progress is allowed for neighbors
            # which were not visited. E.g. if the path is currently A - B,
            # do not consider adding A when iterating over neighbors of B.
            for node in G_undirected.neighbors(last_added):
                if node != second_last_added:
                    if node in path:
                        return False
                    new_path = copy.deepcopy(path)
                    new_path[node] = None
                    queue.append(new_path)

    return True
