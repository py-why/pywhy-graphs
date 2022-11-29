import networkx as nx

from pywhy_graphs.classes.timeseries.base import BaseTimeSeriesDiGraph


def get_summary_graph(G: BaseTimeSeriesDiGraph, include_self_loops: bool = False) -> nx.DiGraph:
    """Compute the summary graph from a time-series graph.

    Parameters
    ----------
    G : BaseTimeSeriesDiGraph
        The time-series graph.

    Returns
    -------
    summary_G : nx.DiGraph
        A possibly cyclic graph.
    """
    # compute the summary graph
    summary_G = nx.DiGraph()

    # add all variables in ts-graph as nodes
    summary_G.add_nodes_from(G.variables)

    # loop through every non-lag node
    for node in G.nodes_at(t=0):
        var_name, _ = node
        for nbr in G.predecessors(node):
            nbr_name, _ = nbr

            # if this is a self-loop, check if we include these
            if not include_self_loops and var_name == nbr_name:
                continue

            # add nbr -> variable
            summary_G.add_edge(nbr_name, var_name)

    return summary_G


def get_extended_summary_graph(G: BaseTimeSeriesDiGraph) -> nx.DiGraph:
    """Compute the extended summary graph from a ts-graph.

    Parameters
    ----------
    G : BaseTimeSeriesDiGraph
        The time-series graph.

    Returns
    -------
    summary_G : nx.DiGraph
        An acyclic extended summary graph with nodes named as tuples of
        (<variable_name>, 't'), or (<variable_name>, '-t') for the present
        and past respectively.
    """
    # compute the summary graph
    summary_G = nx.DiGraph()

    # add all variables in ts-graph as nodes as tuples
    summary_G.add_nodes_from([(variable, "t") for variable in G.variables])
    summary_G.add_nodes_from([(variable, "-t") for variable in G.variables])

    # loop through every non-lag node
    for node in G.nodes_at(t=0):
        var_name, _ = node
        for nbr in G.contemporaneous_neighbors(node):
            nbr_name, _ = nbr

            # add nbr -> variable
            summary_G.add_edge((nbr_name, "t"), (var_name, "t"))

        for nbr in G.lagged_neighbors(node):
            nbr_name, _ = nbr

            # add nbr -> variable
            summary_G.add_edge((nbr_name, "-t"), (var_name, "t"))
    return summary_G
