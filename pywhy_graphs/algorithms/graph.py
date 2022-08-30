from typing import List, Union

import networkx as nx

from pywhy_graphs import ADMG, CPDAG, PAG
from pywhy_graphs.algorithms.generic import _check_adding_cpdag_edge, _check_adding_pag_edge
from pywhy_graphs.typing import Node

__all__ = [
    "is_node_common_cause",
    "set_nodes_as_latent_confounders",
    "is_valid_mec_graph",
]


def is_node_common_cause(G: nx.DiGraph, node: Node, exclude_nodes: List[Node] = None) -> bool:
    """Check if a node is a common cause within the graph.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.
    node : node
        A node in the graph.
    exclude_nodes : list, optional
        Set of nodes to exclude from consideration, by default None.

    Returns
    -------
    is_common_cause : bool
        Whether or not the node is a common cause or not.
    """
    if exclude_nodes is None:
        exclude_nodes = []

    successors = G.successors(node)
    count = 0
    for succ in successors:
        if succ not in exclude_nodes:
            count += 1
        if count >= 2:
            return True
    return False


def set_nodes_as_latent_confounders(G: Union[nx.DiGraph, ADMG], nodes: List[Node]) -> ADMG:
    """Set nodes as latent unobserved confounders.

    Note that this only works if the original node is a common cause
    of some variables in the graph.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.
    nodes : list
        A list of nodes to set. They must all be common causes of
        variables within the graph.

    Returns
    -------
    graph : ADMG
        The mixed-edge causal graph that results.
    """
    bidirected_edges = []
    new_parent_ch_edges = []

    for node in nodes:
        # check if the node is a common cause
        if not is_node_common_cause(G, node, exclude_nodes=nodes):
            raise RuntimeError(
                f"{node} is not a common cause within the graph "
                f"given excluding variables. This function will only convert common "
                f"causes to latent confounders."
            )

        # keep track of which nodes to form c-components over
        successor_nodes = G.successors(node)
        for idx, succ in enumerate(successor_nodes):
            # TODO: do we want this?; add parent -> successor edges
            # if there are parents to this node, they must now point to all the successors
            for parent in G.predecessors(node):
                new_parent_ch_edges.append((parent, succ))

            # form a c-component among the successors
            if idx == 0:
                prev_succ = succ
                continue
            bidirected_edges.append((prev_succ, succ))
            prev_succ = succ

    # create the graph with nodes excluding those that are converted to latent confounders
    if isinstance(G, ADMG):
        new_graph = G.copy()
    elif isinstance(G, nx.DiGraph):
        new_graph = ADMG(G.copy())
    new_graph.remove_nodes_from(nodes)

    # create the c-component structures
    new_graph.add_edges_from(bidirected_edges, new_graph.bidirected_edge_name)

    # add additional edges that need to be accounted for
    new_graph.add_edges_from(new_parent_ch_edges, new_graph.directed_edge_name)
    return new_graph


def is_valid_mec_graph(G: Union[PAG, CPDAG], on_error: str = "raise") -> bool:
    """Check G is a valid PAG.

    A valid CPDAG/PAG is one where each pair of nodes have
    at most one edge between them and the internal graph of directed edges
    do not form cycles.

    Parameters
    ----------
    G : PAG | CPDAG
        The PAG or CPDAG.
    on_error : str
        Whether to raise an error if the graph is non-compliant. Default is 'raise'.
        Other options are 'ignore'.

    Returns
    -------
    bool
        Whether G is a valid PAG or CPDAG.
    """
    if isinstance(G, CPDAG):
        check_func = _check_adding_cpdag_edge
    elif isinstance(G, PAG):
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
