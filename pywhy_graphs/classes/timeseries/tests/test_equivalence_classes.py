import networkx as nx
import pytest

from pywhy_graphs import StationaryTimeSeriesCPDAG


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

    def test_remove_forward_edges(self):
        """Test that a timeseries PAG can remove edges forward in time."""

        # when we remove an edge in a PAG, we should also be able to remove all
        # edges forward in time


class TestTimeSeriesPAG(TimeSeriesMECGraphTester):
    def setup_method(self):
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2; 0 -- 3
        self.Graph = StationaryTimeSeriesCPDAG
        incoming_uncertain_data = [(0, 3)]

        # build dict-of-dict-of-dict K3
        ed2 = {}
        incoming_graph_data = [(0, 1), (0, 2)]
        self.G = self.Graph()
        self.G.add_edges_from(incoming_graph_data, edge_type="directed")
        self.G.add_edges_from(incoming_uncertain_data, edge_type="undirected")


# class TestTimeSeriesPAG(TimeSeriesMECGraphTester):
#     def setup_method(self):
#         # start every graph with the confounded graph
#         # 0 -> 1, 0 -> 2; 0 -- 3
#         self.Graph = StationaryTimeSeriesPAG
#         incoming_uncertain_data = [(0, 3)]

#         # build dict-of-dict-of-dict K3
#         ed2 = {}
#         incoming_graph_data = {0: {1: {}, 2: ed2}}
#         self.G = self.Graph(incoming_graph_data, incoming_uncertain_data)
