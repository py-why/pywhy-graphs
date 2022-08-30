from typing import Union

import networkx as nx

import pywhy_graphs as pgraph
from pywhy_graphs.algorithms.generic import _check_adding_cpdag_edge, _check_adding_pag_edge


def is_valid_mec_graph(G: Union[pgraph.PAG, pgraph.CPDAG], on_error: str = "raise") -> bool:
    """Check G is a valid PAG.

    A valid CPDAG/PAG is one where each pair of nodes have
    at most one edge between them and the internal graph of directed edges
    do not form cycles.

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
    if isinstance(G, pgraph.CPDAG):
        check_func = _check_adding_cpdag_edge
    elif isinstance(G, pgraph.PAG):
        check_func = _check_adding_pag_edge

    for edge_type, edgeview in G.edges().items():
        for u, v in edgeview:
            check_func(G, u, v, edge_type)

    # the directed edges should not form cycles
    if not nx.is_directed_acyclic_graph(G.sub_directed_graph()):
        if on_error == "raise":
            raise RuntimeError(f"{G} is not a DAG, which it should be.")
        return False

    return True
