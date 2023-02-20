import networkx as nx
import numpy as np
import pytest

from pywhy_graphs.classes.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    TimeSeriesDiGraph,
    TimeSeriesGraph,
)
from pywhy_graphs.classes.timeseries.functions import (
    get_extended_summary_graph,
    get_summary_graph,
    has_homologous_edges,
    nodes_in_time_order,
)


@pytest.mark.parametrize(
    "G_func",
    [
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
        TimeSeriesDiGraph,
        TimeSeriesGraph,
    ],
)
def test_has_homologous_edges(G_func):
    max_lag = 3
    G = G_func(max_lag=max_lag)
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", -3), ("x3", 0)),
        (("x1", 0), ("x3", 0)),
    ]
    G.add_edges_from(ts_edges)

    # if graph is stationary, then all time-series edges
    # will have their homologous edges within the graph
    if G.stationary:
        for edge in ts_edges:
            assert has_homologous_edges(G, *edge)
    else:
        # otherwise, not all homologous edges will be within the graph.
        for edge in ts_edges:
            u, v = edge

            # as an edge case, if there are no homologous edges to check, then
            # the function will return True
            if u[1] == -max_lag and v[1] == 0:
                assert has_homologous_edges(G, *edge)
            else:
                assert not has_homologous_edges(G, *edge)


def test_get_summary_graphs():
    max_lag = 2
    ts_edges = [
        (("x2", -1), ("x2", 0)),
        (("x3", -2), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x3", -1), ("x1", 0)),
        (("x3", 0), ("x1", 0)),
        (("x1", -1), ("x1", 0)),
    ]
    G = StationaryTimeSeriesDiGraph(ts_edges, max_lag=max_lag)

    summ_edges = [("x3", "x2"), ("x3", "x1")]

    # check summary graph without self-loops
    expected_summary_G = nx.DiGraph(summ_edges)
    summary_G = get_summary_graph(G)
    assert nx.is_isomorphic(summary_G, expected_summary_G)

    # check summary graph with self-loops
    expected_summary_G.add_edges_from([(node, node) for node in expected_summary_G.nodes])
    summary_G = get_summary_graph(G, include_self_loops=True)
    assert nx.is_isomorphic(summary_G, expected_summary_G)

    with pytest.raises(RuntimeError, match="Undirected graphs not supported"):
        get_summary_graph(G.to_undirected())

    # now test extended summary graph
    ts_edges = [
        (("x2", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x1", 0)),
        (("x3", 0), ("x1", 0)),
        (("x1", -1), ("x1", 0)),
    ]
    expected_ext_summ_G = nx.DiGraph(ts_edges)
    ext_summ_G = get_extended_summary_graph(G)
    assert nx.is_isomorphic(ext_summ_G, expected_ext_summ_G)

    with pytest.raises(RuntimeError, match="Undirected graphs not supported"):
        get_extended_summary_graph(G.to_undirected())


@pytest.mark.parametrize(
    "G_func",
    [
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
        TimeSeriesDiGraph,
        TimeSeriesGraph,
    ],
)
def test_nodes_in_time_order(G_func):
    max_lag = 3
    G = G_func(max_lag=max_lag)
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", -3), ("x3", 0)),
    ]
    G.add_edges_from(ts_edges)
    nodes_set = set(G.nodes)

    nodes = nodes_in_time_order(G)
    current_time = G.max_lag
    for node in nodes:
        assert np.abs(node[1]) <= current_time
        current_time = np.abs(node[1])
        nodes_set.remove(node)
    assert nodes_set == set()
