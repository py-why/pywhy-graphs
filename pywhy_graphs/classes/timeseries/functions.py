from typing import Iterator, List

import networkx as nx
import numpy as np

from pywhy_graphs.classes.timeseries import StationaryTimeSeriesDiGraph, TimeSeriesDiGraph
from pywhy_graphs.typing import Node, TsNode


def nodes_in_time_order(G) -> Iterator:
    """Return nodes from G in time order starting from max-lag to t=0.

    Parameters
    ----------
    G : TimeSeriesGraph
        The timeseries graph.
    """
    for t in range(G.max_lag, -1, -1):
        for node in G.nodes_at(t):
            yield node


def complete_ts_graph(
    variables: List[Node],
    max_lag: int,
    include_contemporaneous: bool = True,
    create_using=StationaryTimeSeriesDiGraph,
):
    """Create a complete time-series graph.

    An analogous function for complete graph from networkx.

    Parameters
    ----------
    variables : list
        List of variables in the time-series graph.
    max_lag : int
        The maximum lag.
    include_contemporaneous : bool, optional
        _description_, by default True
    create_using : func, optional
        The class to create, by default StationaryTimeSeriesDiGraph.

    Returns
    -------
    G : StationaryTimeSeriesDiGraph
        The complete graph.
    """
    G = create_using(max_lag=max_lag)

    # add all possible edges
    for u_node in variables:
        for v_node in variables:
            for u_lag in range(max_lag + 1):
                for v_lag in range(max_lag + 1):
                    if u_lag < v_lag:
                        continue
                    # skip contemporaneous edges if necessary
                    if not include_contemporaneous and u_lag == v_lag:
                        continue
                    # do not add self connections
                    if u_node == v_node and u_lag == v_lag:
                        continue
                    # do not add cyclicity
                    if u_lag == v_lag and (
                        G.has_edge((u_node, -u_lag), (v_node, -v_lag))
                        or G.has_edge((v_node, -v_lag), (u_node, -u_lag))
                    ):
                        continue
                    # if there is already an edge, do not add
                    if G.has_edge((u_node, -u_lag), (v_node, -v_lag)):
                        continue

                    G.add_edge((u_node, -u_lag), (v_node, -v_lag))
    return G


def empty_ts_graph(variables, max_lag, create_using=StationaryTimeSeriesDiGraph):
    G = create_using(max_lag=max_lag)
    for node in variables:
        G.add_node((node, 0))
    return G


def get_summary_graph(G, include_self_loops: bool = False) -> nx.DiGraph:
    """Compute the summary graph from a time-series graph.

    Parameters
    ----------
    G : TimeSeriesDiGraph
        The time-series directed graph. Undirected are not supported.

    Returns
    -------
    summary_G : nx.DiGraph
        A possibly cyclic graph.
    """
    if not G.is_directed():
        raise RuntimeError("Undirected graphs not supported.")

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


def get_extended_summary_graph(G: TimeSeriesDiGraph) -> nx.DiGraph:
    """Compute the extended summary graph from a ts-graph.

    Parameters
    ----------
    G : TimeSeriesDiGraph
        The time-series directed graph. Undirected not supported yet.

    Returns
    -------
    summary_G : nx.DiGraph
        An acyclic extended summary graph with nodes named as tuples of
        (<variable_name>, 0), or (<variable_name>, -1) for the present
        and past respectively.
    """
    if not G.is_directed():
        raise RuntimeError("Undirected graphs not supported.")

    # compute the summary graph
    summary_G = nx.DiGraph()

    # add all variables in ts-graph as nodes as tuples
    summary_G.add_nodes_from([(variable, 0) for variable in G.variables])
    summary_G.add_nodes_from([(variable, -1) for variable in G.variables])

    # loop through every non-lag node
    for node in G.nodes_at(t=0):
        var_name, _ = node
        for nbr in G.contemporaneous_neighbors(node):
            if G.has_edge(node, nbr):
                nbr_name, _ = nbr

                # add nbr -> variable
                summary_G.add_edge((var_name, 0), (nbr_name, 0))
        for nbr in G.lagged_neighbors(node):
            nbr_name, _ = nbr

            # add nbr -> variable
            summary_G.add_edge((nbr_name, -1), (var_name, 0))
    return summary_G


def has_homologous_edges(G, u_of_edge: TsNode, v_of_edge: TsNode) -> bool:
    """Check whether the graph contains all homologous edges for (u, v).

    Parameters
    ----------
    G : time-series graph
        The time-series graph. If the graph is stationary, then it _must_
        have all homologous edges.
    u_of_edge : TsNode
        From node.
    v_of_edge : TsNode
        To node.

    Returns
    -------
    has_edge : bool
        Whether or not the graph contains all homologous edges.

    Notes
    -----
    If the edge is from the max-lag to time point 0, then there are no homologous
    edges to check in the time-series graph. In this case, the function will return
    ``True``.
    """
    u, u_lag = u_of_edge
    v, v_lag = v_of_edge

    u_lag = np.abs(u_lag)
    v_lag = np.abs(v_lag)

    to_t = v_lag
    from_t = u_lag
    for _ in range(u_lag, G._max_lag + 1):
        if not G.has_edge((u, -from_t), (v, -to_t)):
            return False
        to_t += 1
        from_t += 1
    return True
