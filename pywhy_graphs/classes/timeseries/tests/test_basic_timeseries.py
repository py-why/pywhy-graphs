from copy import copy

import networkx as nx
import pytest

from pywhy_graphs.classes.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    TimeSeriesDiGraph,
    TimeSeriesGraph,
    complete_ts_graph,
    has_homologous_edges,
)


@pytest.mark.parametrize(
    "G_func",
    [
        TimeSeriesDiGraph,
        TimeSeriesGraph,
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
def test_ts_graph_error(G_func):
    max_lag = 0
    variables = ["x", "y", "z"]

    with pytest.raises(ValueError, match="Max lag for time series graph "):
        complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)

    G = G_func()
    with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
        G.add_node(0)
    with pytest.raises(ValueError, match="All lag points should be 0, or less."):
        G.add_node(("x", 2))


@pytest.mark.parametrize("G_func", [TimeSeriesGraph, TimeSeriesDiGraph])
def test_time_nodes(G_func):
    """Basic time-series graphs now store nodes as a tuple."""

    G = G_func(max_lag=3)
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


class BaseTimeSeriesNetworkxOperations:
    def test_timeseries_add_node(self):
        max_lag = self.max_lag
        G = self.klass(max_lag=max_lag)

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

        # removal of a node does not remove stationary nodes
        len_nodes = len(G.nodes)
        G.remove_node((1, 0))
        assert len(G.nodes) == len_nodes - 1

        G.remove_variable(1)
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

        G.remove_variables_from([2, 1])
        assert len(G.nodes) == 0

    def test_copy(self):
        max_lag = self.max_lag
        G = self.G

        # copy should retain all edges and structure
        G_copy = G.copy()
        for node in G.variables:
            for lag in range(max_lag + 1):
                assert G_copy.has_node((node, -lag))

        assert nx.is_isomorphic(G, G_copy)


class BaseTimeSeriesGraphTester(BaseTimeSeriesNetworkxOperations):
    """Test basic node and properties of time-series graphs.

    - node/variable querying
    - different types of edges (contemporaneous and lagged)
    - d-separation
    """

    def test_construction(self):
        # test construction directly with edges and by passing
        # in another graph object
        G = self.G.copy()
        new_G = self.klass(G, max_lag=G.max_lag)
        assert nx.is_isomorphic(G, new_G)

    def test_nodes_at(self):
        G = self.G.copy()
        for t in range(G.max_lag + 1):
            nodes = G.nodes_at(t)
            for node in nodes:
                assert node[1] == -t

    def test_contemporaneous_edges(self):
        G = self.G.copy()

        for u, v in G.contemporaneous_edges:
            assert u[1] == v[1]

    def test_lagged_edges(self):
        G = self.G.copy()

        for u, v in G.lag_edges:
            assert v[1] > u[1]
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
        G = G.set_max_lag(max_lag + 1)
        new_nodes = set(G.nodes)
        new_edges = set(G.edges)

        # new nodes should be a superset of the original nodes
        assert all(node in new_nodes for node in nodes)
        assert not all(node in nodes for node in new_nodes)

        if G.stationary:
            # all old edges should have be in the new set of edges
            for edge in edges:
                if not G.has_edge(*edge):
                    assert False
            assert not all(self.G.has_edge(*edge) for edge in new_edges)
        else:
            for edge in edges:
                assert G.has_edge(*edge)
            for edge in new_edges:
                assert self.G.has_edge(*edge)

    def test_d_separation(self):
        # create our own instant
        G_func = self.klass
        max_lag = self.max_lag

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # extend max-lag to double the instance, which is required
        # to check d-separation in stationary graphs
        G_copy = G.copy()
        double_G = G.set_max_lag(2 * G.max_lag)
        # double_G.stationary = False

        if G.is_directed():
            # (x1, -1), (x3, -1) collide at (x2, 0)
            assert nx.is_d_separator(double_G, {("x1", -1)}, {("x3", -1)}, {})
            assert not nx.is_d_separator(double_G, {("x1", -1)}, {("x3", -1)}, {("x2", 0)})

            # (x2, -1), (x2, 0) can be d-separated with lagged points of x1 and x3
            assert nx.is_d_separator(double_G, {("x2", -1)}, {("x2", 0)}, {("x1", -1), ("x3", -1)})

            # we need to all edges going backwards to be able to not d-separate the two
            if G.stationary:
                assert not nx.is_d_separator(double_G, {("x2", -1)}, {("x2", 0)}, {("x3", -1)})
                assert not nx.is_d_separator(double_G, {("x2", -1)}, {("x2", 0)}, {("x1", -1)})

                # note, that d-separation will not work well with max-lag at max-lags
                assert nx.is_d_separator(G_copy, {("x2", -max_lag)}, {("x2", -max_lag + 1)}, {})
                assert not nx.is_d_separator(double_G, {("x2", -max_lag)}, {("x2", -max_lag + 1)}, {})

    def test_add_edge(self):
        max_lag = self.max_lag
        G = self.klass(max_lag=max_lag)

        # test errors with adding edges
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge(1, 2)
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge((1, -2, 3), (1, 0))
        with pytest.raises(ValueError, match="All lag points should be 0, or less"):
            G.add_edge((1, 2), (1, 0))

        # now test adding/removing lagged edges
        # stationarity should be maintained
        G.add_edge((1, -1), (1, 0))
        assert G.has_edge((1, -1), (1, 0))
        to_lag = 0
        if G.stationary:
            for lag in range(1, max_lag):
                assert G.has_edge((1, -lag), (1, -to_lag))
                to_lag = lag
        G.remove_edge((1, -1), (1, 0))
        assert len(G.edges) == 0

        if max_lag > 2:
            G.add_edge((1, -2), (1, 0))
            to_lag = 0
            from_lag = 2
            if G.stationary:
                for lag in range(2, max_lag + 1):
                    assert G.has_edge((1, -from_lag), (1, -to_lag))
                    to_lag += 1
                    from_lag += 1

            G.remove_edge((1, -2), (1, 0))
            assert len(G.edges) == 0

        # now test adding/removing contemporaneous edges
        # stationarity should be maintained
        G.add_edge((2, 0), (1, 0))
        assert G.has_edge((2, 0), (1, 0))
        if G.stationary:
            for lag in range(max_lag):
                assert G.has_edge((2, -lag), (1, -lag))
        G.remove_edge((2, 0), (1, 0))
        assert len(G.edges) == 0

        # when adding edges, the nodes should all be added
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = self.klass(max_lag=max_lag)
        G.add_edges_from(ts_edges)
        for node in ["x1", "x2", "x3"]:
            for lag in range(max_lag + 1):
                assert G.has_node((node, -lag))

    def test_add_edges_from(self):
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        max_lag = self.max_lag
        G = self.klass(max_lag=max_lag)
        G.add_edges_from(ts_edges)
        variables = ("x1", "x2", "x3")
        for var in variables:
            assert var in G.variables

            for lag in range(G.max_lag + 1):
                assert G.has_node((var, -lag))

    def test_remove_homologous_edges(self):
        # create our own instant
        G_func = self.klass
        max_lag = self.max_lag

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        if G.stationary:
            # if the graph is stationary, then after removal of homologous edges
            # we can check for existence
            n_edges = len(G.edges)
            assert has_homologous_edges(G, ("x1", -1), ("x1", 0))
            G.remove_homologous_edges(("x1", -1), ("x1", 0))
            assert not has_homologous_edges(G, ("x1", -1), ("x1", 0))
            assert n_edges - G.max_lag == len(G.edges)
        else:
            # if the graph is nonstationary, there will not be homologous edge structure
            # originally.
            assert not has_homologous_edges(G, ("x1", -1), ("x1", 0))
            orig_G = G.copy()

            # there are no repeating edges similar to the one we have, so
            # the edge count only decreases by one
            G.remove_homologous_edges(("x1", -1), ("x1", 0))
            assert len(orig_G.edges) == len(G.edges) + 1

            orig_G.remove_edge(("x1", -1), ("x1", 0))
            assert nx.is_isomorphic(orig_G, G)

    def test_add_homologous_edges(self):
        # create our own instant
        G_func = self.klass
        max_lag = self.max_lag

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        if G.stationary:
            # if the graph is stationary, then adding homologous edges will add
            # no edges, because the graph is stationary upon construction
            orig_G = G.copy()
            assert has_homologous_edges(G, ("x1", -1), ("x1", 0))
            G.add_homologous_edges(("x1", -1), ("x1", 0))
            assert has_homologous_edges(G, ("x1", -1), ("x1", 0))

            assert nx.is_isomorphic(orig_G, G)
        else:
            # if the graph is nonstationary, there will not be homologous edge structure
            # originally.
            assert not has_homologous_edges(G, ("x1", -1), ("x1", 0))
            orig_G = G.copy()

            # there are no repeating edges similar to the one we have, so
            # the edge count only decreases by one
            G.add_homologous_edges(("x1", -1), ("x1", 0))
            assert has_homologous_edges(G, ("x1", -1), ("x1", 0))
            assert len(orig_G.edges) + G.max_lag - 1 == len(G.edges)

    def test_remove_backwards(self):
        max_lag = self.max_lag
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x2", -1), ("x2", 0)),
            (("x2", -1), ("x1", 0)),
            (("x1", -3), ("x1", 0)),
        ]
        G = self.klass(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # create a copy to compare tests against
        orig_G = G.copy()

        # test backwards removal does not remove unnecessary edges
        last_edge = (("x1", -max_lag), ("x1", -(max_lag - 1)))
        G.remove_homologous_edges(*last_edge, direction="backwards")
        for edge in orig_G.edges:
            if set(edge) != set(last_edge):
                assert edge in G.edges
            else:
                assert edge not in G.edges

        # test backwards removal should remove all backwards edges
        G.add_edge(*last_edge)
        G.remove_homologous_edges(("x1", -1), ("x1", 0), direction="backwards")
        for edge in orig_G.edges:
            u, v = edge
            u_lag = u[1]
            v_lag = v[1]
            if ((u_lag + 1 == v_lag) or (v_lag + 1 == u_lag)) and u[0] == v[0] and u[0] == "x1":
                assert edge not in G.edges
            else:
                assert edge in G.edges

    def test_remove_forwards(self):
        max_lag = self.max_lag
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x2", -1), ("x2", 0)),
            (("x2", -1), ("x1", 0)),
            (("x1", -3), ("x1", 0)),
        ]
        G = self.klass(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # create a copy to compare tests against
        orig_G = G.copy()

        # test forwards removal does not remove unnecessary edges
        first_edge = (("x1", -1), ("x1", 0))
        G.remove_homologous_edges(*first_edge, direction="forward")
        for edge in orig_G.edges:
            if set(edge) == set(first_edge):
                assert edge not in G.edges
            else:
                assert edge in G.edges

        # test forwards removal should remove all forward edges
        G.add_edge(*first_edge)
        last_edge = (("x1", -max_lag), ("x1", -(max_lag - 1)))
        G.remove_homologous_edges(*last_edge, direction="forward")
        for edge in orig_G.edges:
            u, v = edge
            u_lag = u[1]
            v_lag = v[1]
            if ((u_lag + 1 == v_lag) or (v_lag + 1 == u_lag)) and u[0] == v[0] and u[0] == "x1":
                assert edge not in G.edges
            else:
                assert edge in G.edges


class BasicStationaryTimeSeriesGraphTester(BaseTimeSeriesGraphTester):
    def test_nodes_at(self):
        G = self.G.copy()
        for t in range(G.max_lag + 1):
            nodes = G.nodes_at(t)
            for node in nodes:
                assert node[1] == -t

    def test_lag_edges(self):
        G = self.G.copy()

        for u, v in G.lag_edges:
            assert v[1] > u[1]
            assert u[1] < 0

    def test_contemporaneous_edges(self):
        G = self.G.copy()
        for u, v in G.contemporaneous_edges:
            assert u[1] == v[1]

            lag_found = False
            for lag in range(G.max_lag + 1):
                if -lag in (u[1], v[1]):
                    lag_found = True
            if not lag_found:
                assert False

    def test_lagged_nbrs(self):
        G = self.G.copy()

        nbrs = G.lagged_neighbors(("x3", 0))
        assert set(nbrs) == {("x1", -3), ("x3", -1)}


class TestBaseTimesSeriesGraph(BaseTimeSeriesGraphTester):
    def setup_method(self):
        max_lag = 3
        G = TimeSeriesGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
            (("x1", 0), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G
        self.max_lag = max_lag
        self.klass = TimeSeriesGraph


class TestBaseTimesSeriesDiGraph(BaseTimeSeriesGraphTester):
    def setup_method(self):
        max_lag = 3
        G = TimeSeriesDiGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
            (("x1", 0), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G
        self.max_lag = max_lag
        self.klass = TimeSeriesDiGraph


class TestStationaryDiGraph(BasicStationaryTimeSeriesGraphTester):
    """Test properties of a stationary time-series graph."""

    def setup_method(self):
        max_lag = 3
        G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
            (("x1", 0), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G
        self.max_lag = max_lag
        self.klass = StationaryTimeSeriesDiGraph


class TestStationaryGraph(BasicStationaryTimeSeriesGraphTester):
    """Test properties of a stationary time-series graph."""

    def setup_method(self):
        max_lag = 3
        G = StationaryTimeSeriesGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
            (("x1", 0), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G
        self.max_lag = max_lag
        self.klass = StationaryTimeSeriesGraph
