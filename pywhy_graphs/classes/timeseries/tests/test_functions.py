import networkx as nx

from pywhy_graphs import StationaryTimeSeriesDiGraph
from pywhy_graphs.classes.timeseries.functions import get_summary_graph


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
