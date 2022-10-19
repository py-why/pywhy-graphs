from itertools import combinations

import networkx as nx
import pytest

from pywhy_graphs import TimeSeriesGraph
from pywhy_graphs.classes.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    StationaryTimeSeriesMixedEdgeGraph,
    TimeSeriesDiGraph,
    complete_ts_graph,
    empty_ts_graph,
)


@pytest.mark.parametrize(
    "G_func",
    [
        TimeSeriesGraph,
        TimeSeriesDiGraph,
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
def test_ts_graph_error(G_func):
    max_lag = 0
    variables = ["x", "y", "z"]

    with pytest.raises(ValueError, match=""):
        complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)


@pytest.mark.parametrize(
    "max_lag",
    [1, 3],
)
@pytest.mark.parametrize(
    "G_func",
    [
        TimeSeriesGraph,
        TimeSeriesDiGraph,
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
class TestNetworkxIntegration:
    def test_complete_graph(self, G_func, max_lag):
        variables = ["x", "y", "z"]

        if max_lag == 0:
            with pytest.raises(ValueError, match=""):
                complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        else:
            G = complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
            assert G.__class__ == G_func().__class__
            assert set(G.variables) == set(variables)

    def test_empty_graph(self, G_func, max_lag):
        variables = ["x", "y", "z"]

        G = empty_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        assert G.__class__ == G_func().__class__
        assert set(G.variables) == set(variables)
        assert len(G.edges()) == 0

    def test_d_separation(self, G_func, max_lag):
        if issubclass(G_func, nx.Graph):
            return

        variables = ["x", "y", "z"]

        empty_G = empty_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        complete_G = complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        for u, v in combinations(empty_G.nodes, 2):
            assert nx.d_separated(empty_G, {u}, {v}, {})
            assert not nx.d_separated(complete_G, {u}, {v}, {})


@pytest.mark.parametrize(
    "max_lag",
    [1, 3],
)
@pytest.mark.parametrize(
    "G_func",
    [
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
class TestStationaryGraph:
    def test_timeseries_add_node(self, G_func, max_lag):
        G = G_func(max_lag=max_lag)

        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_node(1)
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_node((1, 2, 3))
        with pytest.raises(ValueError, match="All lag points should be 0, or less"):
            G.add_node((1, 2))
        with pytest.raises(ValueError, match="Lag -4 cannot be greater than set max_lag"):
            G.add_node((1, -4))

        # test adding and removing a node
        assert len(G.nodes) == 0
        G.add_node((1, -1))
        assert (1, -1) in G.nodes
        assert (1, 0) in G.nodes
        assert (1, -max_lag) in G.nodes

        # more than the max lag should not be added
        assert (1, -4) not in G.nodes

        G.remove_node((1, -1))
        assert len(G.nodes) == 0

        # test adding and removing multiple nodes
        with pytest.raises(
            ValueError, match=f"Lag {-(max_lag + 1)} cannot be greater than set max_lag"
        ):
            G.add_nodes_from([(1, -1), (1, 0), (1, -(max_lag + 1))])
        G.add_nodes_from([(2, 0), (1, -1), (1, 0), (1, -max_lag)])
        assert len(G.nodes) == 2 * (max_lag + 1)
        assert (1, -1) in G.nodes
        assert (1, 0) in G.nodes
        assert (1, -max_lag) in G.nodes
        assert all((2, -lag) in G.nodes for lag in range(max_lag + 1))

        G.remove_nodes_from([(2, 0), (1, 0)])
        assert len(G.nodes) == 0

    def test_add_edge(self, G_func, max_lag):
        G = G_func(max_lag=max_lag)

        # test errors with adding edges
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge(1, 2)
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge((1, -2, 3), (1, 0))
        with pytest.raises(ValueError, match="All lag points should be 0, or less"):
            G.add_edge((1, 2), (1, 0))
        with pytest.raises(ValueError, match='The lag of the "to" node should be 0'):
            G.add_edge((1, -1), (1, -1))

        # now test adding/removing lagged edges
        # stationarity should be maintained
        G.add_edge((1, -1), (1, 0))
        assert G.has_edge((1, -1), (1, 0))
        to_lag = 0
        for lag in range(1, max_lag):
            assert G.has_edge((1, -lag), (1, -to_lag))
            to_lag = lag
        G.remove_edge((1, -1), (1, 0))
        assert len(G.edges) == 0

        if max_lag > 2:
            G.add_edge((1, -2), (1, 0))
            to_lag = 0
            for lag in range(2, max_lag, 2):
                assert G.has_edge((1, -lag), (1, -to_lag))
                to_lag = lag

            G.remove_edge((1, -2), (1, 0))
            assert len(G.edges) == 0

        # now test adding/removing contemporaneous edges
        # stationarity should be maintained
