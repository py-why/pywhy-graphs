from collections import deque

import networkx as nx
import numpy as np

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
    visited = set()
    all_nodes = graph.nodes()
    G_undirected = graph.get_graphs(edge_type=undirected_edge_name)
    G_directed = graph.get_graphs(edge_type=directed_edge_name)
    # TODO: keep track of paths as first class in queue
    for v in all_nodes:
        print("v:", v)
        seen = {v}
        queue = deque([z for _, z in G_directed.out_edges(nbunch=v)])
        if v in visited:

            continue
        while queue:
            print(queue)
            x = queue.popleft()
            print("pop", x)
            print("seen", seen)
            if x in seen:
                print("appeared in seen", x)
                return False

            seen.add(x)

            for _, node in G_directed.out_edges(nbunch=x):
                print("add out edge", node)
                queue.append(node)
            for nbr in G_undirected.neighbors(x):
                print("add nbr edge", nbr)
                queue.append(nbr)

        visited.add(v)

    return True
