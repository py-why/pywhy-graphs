from itertools import combinations

import networkx as nx
import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.classes.timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    StationaryTimeSeriesMixedEdgeGraph,
    TimeSeriesDiGraph,
    TimeSeriesGraph,
    TimeSeriesMixedEdgeGraph,
)
from pywhy_graphs.classes.timeseries.functions import complete_ts_graph, empty_ts_graph


class TimeSeriesMixedEdgeGraphTester:
    def test_max_lag(self):
        assert self.G.max_lag == self.max_lag

    def test_variables(self):
        assert set(self.G.variables) == set(self.variables)

    def test_copy(self):
        G = self.G.copy()

        for edge_type, graph in G.get_graphs().items():
            assert nx.is_isomorphic(graph, self.G.get_graphs(edge_type))

    def test_ts_graph_errors(self):
        # all subgraph errors should be preserved in this mixed-edge graph
        G = self.G.copy()
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_node("x")
        with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
            G.add_edge((1, -2, 3), (1, 0))
        with pytest.raises(ValueError, match="All lag points should be 0, or less"):
            G.add_edge((1, 2), (1, 0))

    def test_nodes(self):
        G = self.G.copy()
        for variable in G.variables:
            for lag in range(G.max_lag + 1):
                assert G.has_node((variable, -lag))

        # now test adding a node and that all homologous nodes are added
        G.add_node(("newx", 0))
        for lag in range(G.max_lag + 1):
            assert G.has_node(("newx", -lag))

        # now test removing a node and that all homologous nodes are removed
        G.remove_variable("newx")
        for lag in range(G.max_lag + 1):
            assert not G.has_node(("newx", -lag))

    def test_add_edges(self):
        G = self.G.copy()

        # initial graph should have no edges
        for graph in G.get_graphs().values():
            assert set(graph.edges) == set()
        assert ("x", 0) not in G.neighbors(("x", -1))

        # test addition of edges
        G.add_edge(("x", -1), ("x", 0))
        assert ("x", 0) in G.neighbors(("x", -1))
        assert all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

        # only adding edges in a subgraph
        G.remove_edge(("x", -1), ("x", 0), "bidirected")
        G.add_edge(("x", -1), ("x", 0), "directed")
        assert ("x", 0) in G.neighbors(("x", -1))
        assert not all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())
        assert ("x", 0) in G.get_graphs("directed").neighbors(("x", -1))

    def test_remove_edges(self):
        G = self.G.copy()
        G.add_edge(("x", -1), ("x", 0))
        assert ("x", 0) in G.neighbors(("x", -1))
        assert all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

        # now remove the edges
        G.remove_edge(("x", -1), ("x", 0))
        assert ("x", 0) not in G.neighbors(("x", -1))
        assert all(("x", 0) not in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

    @pytest.mark.parametrize("edge_type", ["all", "directed", "bidirected"])
    @pytest.mark.parametrize("direction", ["both", "forward", "backwards"])
    def test_add_homologous_contemporaneous_edges(self, direction, edge_type):
        G = self.G.copy()

        G.add_homologous_edges(("x", -1), ("y", -1), direction=direction, edge_type=edge_type)
        if edge_type == "all":
            edge_type = "any"

        # currently any stationary time series graph adds edges in both directions
        if direction == "both":
            for lag in range(0, G.max_lag + 1):
                assert G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
        elif direction == "forward":
            for lag in range(2, G.max_lag + 1):
                assert not G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
            assert G.has_edge(("x", 0), ("y", 0), edge_type=edge_type)
            assert G.has_edge(("x", -1), ("y", -1), edge_type=edge_type)
        elif direction == "backwards":
            for lag in range(1, G.max_lag + 1):
                assert G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
            assert not G.has_edge(("x", 0), ("y", 0), edge_type=edge_type)

    @pytest.mark.parametrize("edge_type", ["all", "directed", "bidirected"])
    @pytest.mark.parametrize("direction", ["both", "forward", "backwards"])
    def test_remove_homologous_edges(self, direction, edge_type):
        G = self.G.copy()

        G.add_homologous_edges(("x", -1), ("y", -1), direction="both", edge_type=edge_type)
        G.remove_homologous_edges(("x", -1), ("y", -1), direction=direction, edge_type=edge_type)
        if edge_type == "all":
            edge_type = "any"

        # currently any stationary time series graph adds edges in both directions
        if direction == "both":
            for lag in range(0, G.max_lag + 1):
                assert not G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
        elif direction == "forward":
            for lag in range(2, G.max_lag + 1):
                assert G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
            assert not G.has_edge(("x", 0), ("y", 0), edge_type=edge_type)
            assert not G.has_edge(("x", -1), ("y", -1), edge_type=edge_type)
        elif direction == "backwards":
            for lag in range(1, G.max_lag + 1):
                assert not G.has_edge(("x", -lag), ("y", -lag), edge_type=edge_type)
            assert G.has_edge(("x", 0), ("y", 0), edge_type=edge_type)

    def test_m_sep_complete_graph(self):
        variables = self.G.variables
        max_lag = self.G.max_lag

        # test complete graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(self.graph_funcs, self.edge_types):
            graphs.append(
                complete_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)

        G = self.klass(graphs=graphs, edge_types=edge_types, max_lag=max_lag)
        for u, v in combinations(G.nodes, 2):
            assert not pywhy_nx.m_separated(G, {u}, {v}, {})

    def test_m_separation_empty_graph(self):
        G = self.klass(self.graphs, self.edge_types, max_lag=self.max_lag)
        for u, v in combinations(G.nodes, 2):
            assert pywhy_nx.m_separated(self.G, {u}, {v}, {})

    def test_m_separation_confounder(self):
        """Test m-separation with a confounder and collider present."""

        G = self.klass(self.graphs, self.edge_types, max_lag=self.max_lag)

        # create a confounder and selection bias with just directed edges
        directed_edges = [
            (("x", 0), ("y", 0)),
            (("z", 0), ("y", 0)),
            (("x", -1), ("x", 0)),
            (("x", -1), ("z", 0)),
        ]
        G.add_edges_from(directed_edges, edge_type="directed")

        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # without the confounder, the m-separation statements are swapped
        G.remove_edges_from([(("x", -1), ("x", 0)), (("x", -1), ("z", 0))], edge_type="directed")
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # with the latent confounder as a bidirected edge the same m-separation statements hold
        G.add_edge(("x", 0), ("z", 0), edge_type="bidirected")
        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert not pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})

        # without collider on ('y', 0), all three statements are always m-separated
        G.remove_edge(("x", 0), ("z", 0), edge_type="bidirected")
        G.remove_edges_from([(("x", 0), ("y", 0)), (("z", 0), ("y", 0))], edge_type="directed")
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {})
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1)})
        assert pywhy_nx.m_separated(G, {("x", 0)}, {("z", 0)}, {("x", -1), ("y", 0)})


class TestTimeSeriesMixedEdgeGraph(TimeSeriesMixedEdgeGraphTester):
    def setup_method(self):
        max_lag = 3

        variables = ["x", "y", "z"]
        self.graph_funcs = (TimeSeriesDiGraph, TimeSeriesGraph)
        # test empty graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(self.graph_funcs, ("directed", "bidirected")):
            graphs.append(
                empty_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)

        self.graphs = graphs
        self.edge_types = edge_types
        self.klass = TimeSeriesMixedEdgeGraph
        self.variables = variables
        self.max_lag = max_lag
        G = self.klass(self.graphs, self.edge_types, max_lag=self.max_lag)
        self.G = G


class TestStationaryTimeSeriesMixedEdgeGraph(TimeSeriesMixedEdgeGraphTester):
    def setup_method(self):
        max_lag = 3

        variables = ["x", "y", "z"]
        self.graph_funcs = (StationaryTimeSeriesDiGraph, StationaryTimeSeriesGraph)
        # test empty graph
        graphs = []
        edge_types = []
        for graph_func, edge_type in zip(self.graph_funcs, ("directed", "bidirected")):
            graphs.append(
                empty_ts_graph(variables=variables, max_lag=max_lag, create_using=graph_func)
            )
            edge_types.append(edge_type)

        self.graphs = graphs
        self.edge_types = edge_types
        self.klass = StationaryTimeSeriesMixedEdgeGraph
        self.variables = variables
        self.max_lag = max_lag
        G = self.klass(self.graphs, self.edge_types, max_lag=self.max_lag)
        self.G = G

    def test_set_stationarity(self):
        G = self.G.copy()

        G.set_stationarity(True)
        assert G.stationary
        for graph in G.get_graphs().values():
            assert graph.stationary

        G.set_stationarity(False)
        assert not G.stationary
        for graph in G.get_graphs().values():
            assert not graph.stationary
