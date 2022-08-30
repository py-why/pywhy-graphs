from itertools import combinations
from typing import Union

import networkx as nx

import pywhy_graphs as pgraph


def is_valid_mec_graph(G: Union[pgraph.PAG, pgraph.CPDAG], on_error: str = "raise") -> bool:
    """Check G is a valid PAG.

    A valid CPDAG/PAG is one where each pair of nodes have
    at most one edge between them.

    Parameters
    ----------
    G : pgraph.PAG | pgraph.CPDAG
        The PAG or CPDAG.
    on_error : str
        Whether to raise an error if the graph is non-compliant. Default is 'raise'.
        Other options are 'ignore'.

    Returns
    -------
    bool
        Whether G is a valid PAG or CPDAG.
    """
    for node1, node2 in combinations(G.nodes, 2):
        n_edges = 0
        names = []
        for name, graph in G.get_graphs().items():
            if (node1, node2) in graph.edges or (node2, node1) in graph.edges:
                n_edges += 1
                names.append(name)
        if n_edges > 1:
            if on_error == "raise":
                raise RuntimeError(
                    f"There is more than one edge between ({node1}, {node2}) in the "
                    f"edge types: {names}. Please fix the construction of the PAG."
                )
            return False

    # the directed edges should not form cycles
    if not nx.is_directed_acyclic_graph(G.sub_directed_graph()):
        if on_error == "raise":
            raise RuntimeError(f"{G} is not a DAG, which it should be.")
        return False

    return True
