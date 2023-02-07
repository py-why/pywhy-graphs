import logging
from collections import deque

import networkx as nx

import pywhy_graphs.networkx as pywhy_nx

__all__ = ["m_separated"]

logger = logging.getLogger(__name__)


def m_separated(
    G,
    x,
    y,
    z,
    bidirected_edge_name="bidirected",
    directed_edge_name="directed",
    undirected_edge_name="undirected",
):
    """Check m-separation among 'x' and 'y' given 'z' in mixed-edge causal graph G, which may
    contain directed, bidirected, and undirected edges.

    This implements the m-separation algorithm TESTSEP presented in [1]_, with additional
    checks to ensure that it works for non-ancestral mixed graphs (e.g. ADMGs).


    Parameters
    ----------
    G : mixed-edge-graph
        Mixed edge causal graph.
    x : set
        First set of nodes in ``G``.
    y : set
        Second set of nodes in ``G``.
    z : set
        Set of conditioning nodes in ``G``. Can be empty set.
    directed_edge_name : str
        Name of the directed edge, default is directed.
    bidirected_edge_name : str
        Name of the bidirected edge, default is bidirected.
    undirected_edge_name : str
        Name of the undirected edge, default is undirected.

    Returns
    -------
    b : bool
        A boolean that is true if ``x`` is m-separated from ``y`` given ``z`` in ``G``.

    References
    ----------
    .. [1] B. van der Zander, M. Liśkiewicz, and J. Textor, “Separators and Adjustment
       Sets in Causal Graphs: Complete Criteria and an Algorithmic Framework,” Artificial
       Intelligence, vol. 270, pp. 1–40, May 2019, doi: 10.1016/j.artint.2018.12.006.

    .. [2] Spirtes, P. and Richardson, T.S.. (1997). A Polynomial Time Algorithm
       for Determining DAG Equivalence in the Presence of Latent Variables and Selection
       Bias. Proceedings of the Sixth International Workshop on Artificial Intelligence and
       Statistics, in Proceedings of Machine Learning Research

    See Also
    --------
    networkx.algorithms.d_separation.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs and does
    not have an ``ADMG`` representation.
    """
    if not isinstance(G, pywhy_nx.MixedEdgeGraph):
        raise nx.NetworkXError(
            "m-separation should only be run on a MixedEdgeGraph. If "
            'you have a directed graph, use "d_separated" function instead.'
        )
    if not set(G.edge_types).issubset(
        {directed_edge_name, bidirected_edge_name, undirected_edge_name}
    ):
        raise nx.NetworkXError(
            f"m-separation only works on graphs with directed, bidirected, and undirected edges. "
            f"Your graph passed in has the following edge types: {G.edge_types}, whereas "
            f"the function is expecting directed edges named {directed_edge_name}, "
            f"bidirected edges named {bidirected_edge_name}, and undirected edges "
            f"named {undirected_edge_name}."
        )

    if directed_edge_name in G.edge_types:
        if not nx.is_directed_acyclic_graph(G.get_graphs(directed_edge_name)):
            raise nx.NetworkXError("directed edge graph should be directed acyclic")

    # contains -> and <-> edges from starting node T
    forward_deque = deque([])
    forward_visited = set()

    # contains <- and - edges from starting node T
    backward_deque = deque(x)
    backward_visited = set()
    has_undirected = undirected_edge_name in G.edge_types
    if has_undirected:
        G_undirected = G.get_graphs(edge_type=undirected_edge_name)
    has_directed = directed_edge_name in G.edge_types

    an_z = z
    if has_directed:
        G_directed = G.get_graphs(edge_type=directed_edge_name)
        an_z = set().union(*[nx.ancestors(G_directed, x) for x in z]).union(z)

    has_bidirected = bidirected_edge_name in G.edge_types
    if has_bidirected:
        G_bidirected = G.get_graphs(edge_type=bidirected_edge_name)

    while forward_deque or backward_deque:

        if backward_deque:
            node = backward_deque.popleft()
            backward_visited.add(node)
            if node in y:
                return False
            if node in z:
                continue

            # add - edges to forward deque
            if has_undirected:
                for nbr in G_undirected.neighbors(node):
                    if nbr not in backward_visited:
                        backward_deque.append(nbr)

            if has_directed:
                # add <- edges to backward deque
                for x, _ in G_directed.in_edges(nbunch=node):
                    if x not in backward_visited:
                        backward_deque.append(x)

                # add -> edges to forward deque
                for _, x in G_directed.out_edges(nbunch=node):
                    if x not in forward_visited:
                        forward_deque.append(x)

            # add <-> edge to forward deque
            if has_bidirected:
                for nbr in G_bidirected.neighbors(node):
                    if nbr not in forward_visited:
                        forward_deque.append(nbr)

        if forward_deque:
            node = forward_deque.popleft()
            forward_visited.add(node)
            if node in y:
                return False

            # Consider if *-> node <-* is opened due to conditioning on collider,
            # or descendant of collider
            if node in an_z:

                if has_directed:
                    # add <- edges to backward deque
                    for x, _ in G_directed.in_edges(nbunch=node):
                        if x not in backward_visited:
                            backward_deque.append(x)

                # add <-> edge to backward deque
                if has_bidirected:
                    for nbr in G_bidirected.neighbors(node):
                        if nbr not in forward_visited:
                            forward_deque.append(nbr)

            if node not in z:
                if has_undirected:
                    for nbr in G_undirected.neighbors(node):
                        if nbr not in backward_visited:
                            backward_deque.append(nbr)

                if has_directed:
                    # add -> edges to forward deque
                    for _, x in G_directed.out_edges(nbunch=node):
                        if x not in forward_visited:
                            forward_deque.append(x)

    return True
