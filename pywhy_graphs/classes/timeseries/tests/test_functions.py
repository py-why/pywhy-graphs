import networkx as nx
import numpy as np
import pytest

from pywhy_graphs import StationaryTimeSeriesDiGraph
from pywhy_graphs.classes.timeseries.functions import get_summary_graph
from pywhy_graphs.classes.timeseries.timeseries import (
    BaseTimeSeriesDiGraph,
    BaseTimeSeriesGraph,
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    nodes_in_time_order,
)


def test_get_summary_graph():
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


@pytest.mark.parametrize(
    "G_func",
    [
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
        BaseTimeSeriesDiGraph,
        BaseTimeSeriesGraph,
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
