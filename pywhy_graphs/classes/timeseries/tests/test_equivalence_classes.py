import networkx as nx
import pytest

from pywhy_graphs import StationaryTimeSeriesCPDAG, StationaryTimeSeriesPAG


class TimeSeriesMECGraphTester:
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
        G = self.Graph(max_lag=self.max_lag)

        # initial graph should have no edges
        for graph in G.get_graphs().values():
            assert set(graph.edges) == set()
        G.add_edge(("x", 0), ("y", 0))
        assert ("x", 0) not in G.neighbors(("x", -1))

        # test addition of edges
        G.add_edge(("x", -1), ("x", 0))
        assert ("x", 0) in G.neighbors(("x", -1))
        assert all(("x", 0) in graph.neighbors(("x", -1)) for graph in G.get_graphs().values())

        # only adding edges in a subgraph
        G.remove_edge(("x", -1), ("x", 0), "undirected")
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

    def test_remove_forward_edges(self):
        """Test that a timeseries PAG can remove edges forward in time."""

        # when we remove an edge in a PAG, we should also be able to remove all
        # edges forward in time


class TestTimeSeriesCPDAG(TimeSeriesMECGraphTester):
    def setup_method(self):
        # start with a single time-series and graph over lags
        self.Graph = StationaryTimeSeriesCPDAG
        self.max_lag = 3
        incoming_uncertain_data = [((0, -3), (0, -2))]

        # build dict-of-dict-of-dict K3
        incoming_graph_data = [((0, -2), (0, -1))]
        self.G = self.Graph(max_lag=self.max_lag)
        self.G.add_edges_from(incoming_graph_data, edge_type="directed")
        self.G.add_edges_from(incoming_uncertain_data, edge_type="undirected")

        self.k3nodes = self.G.nodes
        self.variables = [0]


class TestTimeSeriesPAG(TimeSeriesMECGraphTester):
    def setup_method(self):
        # start with a single time-series and graph over lags
        self.Graph = StationaryTimeSeriesPAG
        self.max_lag = 3
        incoming_uncertain_data = [((0, -3), (0, -2))]

        # build dict-of-dict-of-dict K3
        incoming_graph_data = [((0, -2), (0, -1))]
        self.G = self.Graph(max_lag=self.max_lag)
        self.G.add_edges_from(incoming_graph_data, edge_type="directed")
        self.G.add_edges_from(incoming_uncertain_data, edge_type="undirected")

        self.k3nodes = self.G.nodes
        self.variables = [0]
