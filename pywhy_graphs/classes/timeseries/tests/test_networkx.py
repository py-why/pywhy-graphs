"""Test basic functions that have an analogous version in networkx."""

from itertools import combinations

import networkx as nx
import pytest

from pywhy_graphs.classes.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    TimeSeriesDiGraph,
    TimeSeriesGraph,
)
from pywhy_graphs.classes.timeseries.functions import complete_ts_graph, empty_ts_graph


@pytest.mark.parametrize(
    "max_lag",
    [1, 3],
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
class TestNetworkxIntegration:
    """Test core-networkx-like functions.

    - complete_graph
    - empty_graph
    - d_separated
    """

    def test_complete_graph(self, G_func, max_lag):
        variables = ["x", "y", "z"]

        if max_lag == 0:
            with pytest.raises(ValueError, match=""):
                complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        else:
            G = complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
            assert G.__class__ == G_func().__class__
            assert set(G.variables) == set(variables)

            for u, v in combinations(variables, 2):
                for u_lag in range(max_lag + 1):
                    for v_lag in range(max_lag + 1):
                        if u_lag < v_lag:
                            continue
                        assert ((u, -u_lag), (v, -v_lag)) in G.edges()

    def test_empty_graph(self, G_func, max_lag):
        variables = ["x", "y", "z"]

        G = empty_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        assert G.__class__ == G_func().__class__
        assert set(G.variables) == set(variables)
        assert len(G.edges()) == 0

    def test_d_separation(self, G_func, max_lag):
        if issubclass(G_func, nx.Graph):
            return
        if issubclass(G_func, nx.MixedEdgeGraph):
            return

        variables = ["x", "y", "z"]

        empty_G = empty_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        complete_G = complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)
        for u, v in combinations(empty_G.nodes, 2):
            assert nx.is_d_separator(empty_G, {u}, {v}, {})
            assert not nx.is_d_separator(complete_G, {u}, {v}, {})
