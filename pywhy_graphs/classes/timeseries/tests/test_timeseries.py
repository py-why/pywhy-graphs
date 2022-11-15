from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from pywhy_graphs.classes.timeseries.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    StationaryTimeSeriesMixedEdgeGraph,
    complete_ts_graph,
    empty_ts_graph,
    nodes_in_time_order,
)


@pytest.mark.parametrize("G_func", [StationaryTimeSeriesGraph, StationaryTimeSeriesDiGraph])
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


@pytest.mark.parametrize(
    "G_func",
    [
        StationaryTimeSeriesGraph,
        StationaryTimeSeriesDiGraph,
    ],
)
def test_ts_graph_error(G_func):
    max_lag = 0
    variables = ["x", "y", "z"]

    with pytest.raises(ValueError, match="Max lag for time series graph "):
        complete_ts_graph(variables=variables, max_lag=max_lag, create_using=G_func)


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
            assert nx.d_separated(empty_G, {u}, {v}, {})
            assert not nx.d_separated(complete_G, {u}, {v}, {})


class TestStationaryDiGraphProperties:
    """Test properties of a stationary time-series graph."""

    def setup(self):
        max_lag = 3
        G = StationaryTimeSeriesGraph(max_lag=max_lag)
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
            (("x1", -3), ("x3", 0)),
        ]
        G.add_edges_from(ts_edges)
        self.G = G

    def test_nodes_at(self):
        G = self.G.copy(double_max_lag=False)
        for t in range(G.max_lag + 1):
            nodes = G.nodes_at(t)
            for node in nodes:
                assert node[1] == -t

    def test_contemporaneous_edge(self):
        G = self.G.copy(double_max_lag=False)

        for u, v in G.contemporaneous_edges:
            assert u[1] == v[1] == 0

    def test_lag_edges(self):
        G = self.G.copy(double_max_lag=False)

        for u, v in G.lag_edges:
            assert v[1] == 0
            assert u[1] < 0

    def test_lagged_nbrs(self):
        G = self.G.copy(double_max_lag=False)

        nbrs = G.lagged_neighbors(("x3", 0))
        assert set(nbrs) == {("x1", -3), ("x3", -1)}

    def test_contemporaneous_nbrs(self):
        G = self.G.copy(double_max_lag=False)

        nbrs = G.contemporaneous_neighbors(("x3", 0))
        assert set(nbrs) == set()


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
    """Test stationary graph adding and removing nodes and edges.

    Stationary graphs add homologous edges every time they add edges.
    During removal of edges, they can remove homologous edges either altogether,
    or by default
    """

    def test_construction(self, G_func, max_lag):
        # test construction directly with edges and by passing
        # in another graph object
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(ts_edges, max_lag=max_lag)
        new_G = G_func(G, max_lag=max_lag)
        assert nx.is_isomorphic(G, new_G)

    def test_timeseries_add_node(self, G_func, max_lag):
        G = G_func(max_lag=max_lag)

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
            from_lag = 2
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
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)
        for node in ["x1", "x2", "x3"]:
            for lag in range(max_lag + 1):
                assert G.has_node((node, -lag))

    def test_add_edges_from(self, G_func, max_lag):
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)
        variables = ("x1", "x2", "x3")
        for var in variables:
            assert var in G.variables

            for lag in range(G.max_lag + 1):
                assert G.has_node((var, -lag))

    def test_copy(self, G_func, max_lag):
        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x1", -1), ("x2", 0)),
            (("x3", -1), ("x2", 0)),
            (("x3", -1), ("x3", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # copy should retain all edges and structure
        G_copy = G.copy(double_max_lag=False)
        for node in ["x1", "x2", "x3"]:
            for lag in range(max_lag + 1):
                assert G_copy.has_node((node, -lag))

        for u, nbrs in G._adj.items():
            for v, datadict in nbrs.items():
                print(u, v, datadict)
        print(G.nodes)
        print(G_copy.nodes)
        assert nx.is_isomorphic(G, G_copy)

        if isinstance(G, StationaryTimeSeriesDiGraph):
            assert nx.d_separated(G_copy, {("x2", -1)}, {("x2", 0)}, {("x1", -1), ("x3", -1)})
            assert nx.d_separated(G_copy, {("x2", 0)}, {("x2", -1)}, {("x1", -1), ("x3", -1)})
            assert not nx.d_separated(G, {("x2", -1)}, {("x2", 0)}, {})

    def test_remove_backwards(self, G_func, max_lag):
        if max_lag < 3:
            return

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x2", -1), ("x2", 0)),
            (("x2", -1), ("x1", 0)),
            (("x1", -3), ("x1", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # create a copy to compare tests against
        orig_G = G.copy(double_max_lag=False)

        with pytest.raises(ValueError, match="Auto removal should be one"):
            G.set_auto_removal(False)
        with pytest.raises(ValueError, match="Auto removal should be one"):
            G.set_auto_removal(True)

        # test backwards removal does not remove unnecessary edges
        G.set_auto_removal("backwards")
        last_edge = (("x1", -max_lag), ("x1", -(max_lag - 1)))
        G.remove_edge(*last_edge)
        for edge in orig_G.edges:
            if set(edge) != set(last_edge):
                assert edge in G.edges
            else:
                assert edge not in G.edges

        # test backwards removal should remove all backwards edges
        G.add_edge(*last_edge)
        G.remove_edge(("x1", -1), ("x1", 0))
        for edge in orig_G.edges:
            u, v = edge
            u_lag = u[1]
            v_lag = v[1]
            if ((u_lag + 1 == v_lag) or (v_lag + 1 == u_lag)) and u[0] == v[0] and u[0] == "x1":
                assert edge not in G.edges
            else:
                assert edge in G.edges

    def test_remove_forwards(self, G_func, max_lag):
        if max_lag < 3:
            return

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x2", -1), ("x2", 0)),
            (("x2", -1), ("x1", 0)),
            (("x1", -3), ("x1", 0)),
        ]
        G = G_func(max_lag=max_lag)
        G.add_edges_from(ts_edges)

        # create a copy to compare tests against
        orig_G = G.copy(double_max_lag=False)

        G.set_auto_removal("forwards")

        # test forwards removal does not remove unnecessary edges
        first_edge = (("x1", -1), ("x1", 0))
        G.remove_edge(*first_edge)
        for edge in orig_G.edges:
            if set(edge) == set(first_edge):
                assert edge not in G.edges
            else:
                assert edge in G.edges

        # test forwards removal should remove all forward edges
        G.add_edge(*first_edge)
        last_edge = (("x1", -max_lag), ("x1", -(max_lag - 1)))
        G.remove_edge(*last_edge)
        for edge in orig_G.edges:
            u, v = edge
            u_lag = u[1]
            v_lag = v[1]
            if ((u_lag + 1 == v_lag) or (v_lag + 1 == u_lag)) and u[0] == v[0] and u[0] == "x1":
                assert edge not in G.edges
            else:
                assert edge in G.edges

    def test_remove_none(self, G_func, max_lag):
        if max_lag < 3:
            return

        ts_edges = [
            (("x1", -1), ("x1", 0)),
            (("x2", -1), ("x2", 0)),
            (("x2", -1), ("x1", 0)),
            (("x1", -3), ("x1", 0)),
        ]
        G = G_func(max_lag=max_lag)
        print(G)
        print(G.graph)
        G.add_edges_from(ts_edges)

        # create a copy to compare tests against
        orig_G = G.copy(double_max_lag=False)

        G.set_auto_removal(None)

        # test forwards removal does not remove unnecessary edges
        first_edge = (("x1", -1), ("x1", 0))
        G.remove_edge(*first_edge)
        for edge in orig_G.edges:
            if set(edge) == set(first_edge):
                assert edge not in G.edges
            else:
                assert edge in G.edges

        # test forwards removal should remove all forward edges
        G.add_edge(*first_edge)
        last_edge = (("x1", -max_lag), ("x1", -(max_lag - 1)))
        G.remove_edge(*last_edge)
        for edge in orig_G.edges:
            if set(edge) == set(last_edge):
                assert edge not in G.edges
            else:
                assert edge in G.edges


class TestStationaryMixedEdgeGraph:
    def setup(self):
        max_lag = 3

        variables = ["x", "y", "z"]
        # test empty graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(
            (StationaryTimeSeriesDiGraph, StationaryTimeSeriesGraph), ("directed", "bidirected")
        ):
            graphs.append(
                empty_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)
        G = StationaryTimeSeriesMixedEdgeGraph(
            graphs=graphs, edge_types=edge_types, max_lag=max_lag
        )
        for u, v in combinations(G.nodes, 2):
            assert nx.m_separated(G, {u}, {v}, {})

        self.G = G
        self.variables = variables
        self.max_lag = max_lag

    def test_max_lag(self):
        assert self.G.max_lag == self.max_lag

    def test_variables(self):
        assert set(self.G.variables) == set(self.variables)

    def test_copy(self):
        G = self.G.copy(double_max_lag=False)

        for edge_type, graph in G.get_graphs().items():
            assert nx.is_isomorphic(graph, self.G.get_graphs(edge_type))

    def test_ts_graph_errors(self):
        # all subgraph errors should be preserved in this mixed-edge graph
        G = self.G.copy(double_max_lag=False)
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_node("x")
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge((1, -2, 3), (1, 0))
        with pytest.raises(ValueError, match="All lag points should be 0, or less"):
            G.add_edge((1, 2), (1, 0))

    def test_nodes(self):
        G = self.G.copy(double_max_lag=False)
        G.set_auto_removal("backwards")
        for variable in G.variables:
            for lag in range(G.max_lag + 1):
                assert G.has_node((variable, -lag))

        # now test adding a node and that all homologous nodes are added
        G.add_node(("newx", 0))
        for lag in range(G.max_lag + 1):
            assert G.has_node(("newx", -lag))

        # now test removing a node and that all homologous nodes are removed
        G.remove_node(("newx", 0))
        for lag in range(G.max_lag + 1):
            assert not G.has_node(("newx", -lag))

    def test_edges(self):
        G = self.G.copy(double_max_lag=False)

        # initial graph should have no edges
        for graph in G.get_graphs().values():
            assert set(graph.edges) == set()

        assert ("x", 0) not in G.neighbors(("x", -1))

        # test addition of edges
        G.add_edge(("x", -1), ("x", 0))
        assert ("x", 0) in G.neighbors(("x", -1))
        assert all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

        # now remove the edges
        G.remove_edge(("x", -1), ("x", 0))
        assert ("x", 0) not in G.neighbors(("x", -1))
        assert all(("x", 0) not in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

        # only adding edges in a subgraph
        G.add_edge(("x", -1), ("x", 0), "directed")
        assert ("x", 0) in G.neighbors(("x", -1))
        assert not all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())
        assert ("x", 0) in G.get_graphs("directed").neighbors(("x", -1))


class TestStationaryMixedEdgeGraphMSep:
    def setup(self):
        max_lag = 3

        variables = ["x", "y", "z"]
        # test empty graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(
            (StationaryTimeSeriesDiGraph, StationaryTimeSeriesGraph), ("directed", "bidirected")
        ):
            graphs.append(
                empty_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)
        G = StationaryTimeSeriesMixedEdgeGraph(
            graphs=graphs, edge_types=edge_types, max_lag=max_lag
        )
        for u, v in combinations(G.nodes, 2):
            assert nx.m_separated(G, {u}, {v}, {})

        self.G = G
        self.variables = variables
        self.max_lag = max_lag

    def test_m_separation_complete_graph(self):
        variables = self.G.variables
        max_lag = self.G.max_lag

        # test complete graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(
            (StationaryTimeSeriesDiGraph, StationaryTimeSeriesGraph), ("directed", "bidirected")
        ):
            graphs.append(
                complete_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)

        G = StationaryTimeSeriesMixedEdgeGraph(
            graphs=graphs, edge_types=edge_types, max_lag=max_lag
        )
        for u, v in combinations(G.nodes, 2):
            assert not nx.m_separated(G, {u}, {v}, {})

    def test_m_separation_empty_graph(self):
        for u, v in combinations(self.G.nodes, 2):
            assert nx.m_separated(self.G, {u}, {v}, {})

    def test_m_separation_confounder(self):
        """Test m-separation with a confounder and collider present."""

        G = self.G.copy(double_max_lag=False)
        G.set_auto_removal("backwards")

        # create a confounder and selection bias with just directed edges
        directed_edges = [
            (("x", 0), ("y", 0)),
            (("z", 0), ("y", 0)),
            (("x", -1), ("x", 0)),
            (("x", -1), ("z", 0)),
        ]
        G.add_edges_from(directed_edges, edge_type="directed")

        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # without the confounder, the m-separation statements are swapped
        G.remove_edges_from([(("x", -1), ("x", 0)), (("x", -1), ("z", 0))], edge_type="directed")
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # with the latent confounder as a bidirected edge the same m-separation statements hold
        G.add_edge(("x", 0), ("z", 0), edge_type="bidirected")
        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # without collider on ('y', 0), all three statements are always m-separated
        G.remove_edge(("x", 0), ("z", 0), edge_type="bidirected")
        G.remove_edges_from([(("x", 0), ("y", 0)), (("z", 0), ("y", 0))], edge_type="directed")
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})
