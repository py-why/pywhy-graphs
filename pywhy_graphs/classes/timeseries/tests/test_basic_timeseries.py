from copy import copy

import pytest

from pywhy_graphs.classes.timeseries import (
    BaseTimeSeriesDiGraph,
    BaseTimeSeriesGraph,
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    complete_ts_graph,
)


@pytest.mark.parametrize(
    "G_func",
    [
        BaseTimeSeriesDiGraph,
        BaseTimeSeriesGraph,
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
def test_ts_graph_error(G_func):
    max_lag = 0
    variables = ["x", "y", "z"]

    with pytest.raises(ValueError, match="Max lag for time series graph "):
        complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)


@pytest.mark.parametrize("G_func", [BaseTimeSeriesGraph, BaseTimeSeriesDiGraph])
def test_time_nodes(G_func):
    """Basic time-series graphs now store nodes as a tuple."""

    G = G_func()
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", -3), ("x3", 0)),
    ]
    G.add_edges_from(ts_edges)

    for node in G.nodes:
        assert len(node) == 2
        assert node[1] <= 0

    with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
        G.add_node(1)
    with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
        G.add_node((1, 2, 3))


class BaseTimeSeriesGraphTester:
    """Test basic node and properties of time-series graphs."""

    def test_nodes_at(self):
        G = self.G.copy()
        for t in range(G.max_lag + 1):
            nodes = G.nodes_at(t)
            for node in nodes:
                assert node[1] == -t

    def test_contemporaneous_edge(self):
        G = self.G.copy()

        for u, v in G.contemporaneous_edges:
            assert u[1] == v[1] == 0

    def test_lag_edges(self):
        G = self.G.copy()

        for u, v in G.lag_edges:
            assert v[1] == 0
            assert u[1] < 0

    def test_lagged_nbrs(self):
        G = self.G.copy()
        G.add_edge(("x1", -3), ("x3", 0))
        G.add_edge(("x3", -1), ("x3", 0))

        nbrs = G.lagged_neighbors(("x3", 0))
        assert set(nbrs) == {("x1", -3), ("x3", -1)}

    def test_contemporaneous_nbrs(self):
        G = self.G.copy()

        G.add_node(("new_node", 0))
        nbrs = G.contemporaneous_neighbors(("new_node", 0))
        assert set(nbrs) == set()

    def test_set_max_lag(self):
        G = self.G.copy()

        # get the current max lag
        max_lag = copy(self.G.max_lag)

        # get the current nodes and edges
        nodes = set(self.G.nodes)
        edges = set(self.G.edges)

        # first set max-lag higher
        G.set_max_lag(max_lag + 1)
        new_nodes = set(G.nodes)
        new_edges = set(G.edges)

        # new nodes should be a superset of the original nodes
        assert all(node in new_nodes for node in nodes)
        assert not all(node in nodes for node in new_nodes)

        if G.stationary:
            assert all(edge in new_edges for edge in edges)
            assert not all(edge in edges for edge in new_edges)
        else:
            for edge in edges:
                assert G.has_edge(*edge)
            for edge in new_edges:
                assert self.G.has_edge(*edge)


class TestBaseTimesSeriesGraph(BaseTimeSeriesGraphTester):
    def setup(self):
        max_lag = 3
        G = BaseTimeSeriesGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G


class TestBaseTimesSeriesDiGraph(BaseTimeSeriesGraphTester):
    def setup(self):
        max_lag = 3
        G = BaseTimeSeriesDiGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G
