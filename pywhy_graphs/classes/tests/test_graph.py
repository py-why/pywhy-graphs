import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs import ADMG, CPDAG, PAG


class BaseGraph:
    def test_children_and_parents(self):
        """Test working with children and parents."""
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.G.copy()

        # basic parent/children semantics
        assert [1, 2] == list(G.children(0))
        assert [] == list(G.parents(0))
        assert [] == list(G.children(1))
        assert [0] == list(G.parents(1))

        # a lone bidirected edge is not a child or a parent
        G.add_edge(2, 3, "undirected")
        assert [] == list(G.parents(3))
        assert [] == list(G.children(3))

    def test_size(self):
        G = self.G

        # size stores all edges
        assert G.number_of_edges() == 3
        assert G.number_of_edges(edge_type="directed") == 2
        # TODO: size() does not work yet due to degree
        # assert G.size() == 3


class TestCPDAG(BaseGraph):
    def setup_method(self):
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2; 0 -- 3
        self.Graph = CPDAG
        incoming_uncertain_data = [(0, 3)]

        # build dict-of-dict-of-dict K3
        ed2 = {}
        incoming_graph_data = {0: {1: {}, 2: ed2}}
        self.G = self.Graph(incoming_graph_data, incoming_uncertain_data)


class TestADMG(BaseGraph):
    """Test relevant causal graph properties."""

    def setup_method(self):
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        self.Graph = ADMG
        incoming_latent_data = [(0, 1)]

        # build dict-of-dict-of-dict K3
        ed1, ed2 = ({}, {})
        incoming_graph_data = {0: {1: ed1, 2: ed2}}
        self.G = self.Graph(incoming_graph_data, incoming_latent_data)

    def test_bidirected_edge(self):
        """Test bidirected edge functions."""
        # add bidirected edge to an isolated node
        G = self.G.copy()

        # error if edge types are not specified correctly
        # in adherence with networkx API
        with pytest.raises(ValueError, match="Edge type bi-directed not"):
            G.add_edge(1, 5, edge_type="bi-directed")

        # 'bidirected' is a default edge type name in ADMG
        G.add_edge(1, 5, "bidirected")
        assert G.has_edge(1, 5, "bidirected")
        assert G.has_edge(5, 1, "bidirected")
        G.remove_edge(1, 5, "bidirected")
        assert 5 in G
        assert not G.has_edge(1, 5, "bidirected")
        assert not G.has_edge(5, 1, "bidirected")
        # TODO: make work once degree works as expected
        # assert nx.is_isolate(G, 5)

    def test_c_components(self):
        """Test working with c-components in causal graph."""
        G = self.G
        print(self.G)
        print(self.G.edges())

        assert list(G.c_components()) == [{0, 1}, {2}]

    # def test_hash(self):
    #     """Test hashing a causal graph."""
    #     G = self.G
    #     current_hash = hash(G)
    #     assert G._current_hash is None

    #     G.add_bidirected_edge("1", "2")
    #     new_hash = hash(G)
    #     assert current_hash != new_hash

    #     G.remove_bidirected_edge("1", "2")
    #     assert current_hash == hash(G)

    # def test_full_graph(self):
    #     """Test computing a full graph from causal graph."""
    #     G = self.G
    #     # the current hash should match after computing full graphs
    #     current_hash = hash(G)
    #     G.compute_full_graph()
    #     assert current_hash == G._current_hash
    #     G.compute_full_graph()
    #     assert current_hash == G._current_hash

    #     # after adding a new edge, the hash should change and
    #     # be different
    #     G.add_bidirected_edge("1", "2")
    #     new_hash = hash(G)
    #     assert new_hash != G._current_hash

    #     # once the hash is computed, it should be the same again
    #     G.compute_full_graph()
    #     assert new_hash == G._current_hash

    #     # removing the bidirected edge should result in the same
    #     # hash again
    #     G.remove_bidirected_edge("1", "2")
    #     assert current_hash != G._current_hash
    #     G.compute_full_graph()
    #     assert current_hash == G._current_hash

    #     # different orders of edges shouldn't matter
    #     G_copy = G.copy()
    #     G.add_bidirected_edge("1", "2")
    #     G.add_bidirected_edge("2", "3")
    #     G_hash = hash(G)
    #     G_copy.add_bidirected_edge("2", "3")
    #     G_copy.add_bidirected_edge("1", "2")
    #     copy_hash = hash(G_copy)
    #     assert G_hash == copy_hash

    def test_m_separation(self):
        G = self.G.copy()
        # add collider on 0
        G.add_edge(3, 0, G.directed_edge_name)

        # normal d-separation statements should hold
        assert not pywhy_nx.m_separated(G, {1}, {2}, set())
        assert pywhy_nx.m_separated(G, {1}, {2}, {0})

        # when we add an edge from 0 -> 1
        # there is no d-separation statement
        assert not pywhy_nx.m_separated(G, {3}, {1}, set())
        assert not pywhy_nx.m_separated(G, {3}, {1}, {0})

        # test collider works on bidirected edge
        # 1 <-> 0
        G.remove_edge(0, 1, G.directed_edge_name)
        assert pywhy_nx.m_separated(G, {3}, {1}, set())
        assert not pywhy_nx.m_separated(G, {3}, {1}, {0})


class TestPAG(TestADMG):
    def setup_method(self):
        # setup the causal graph in previous method
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        super().setup_method()
        self.Graph = PAG
        self.G = PAG(self.G.sub_directed_graph())

        # Create a PAG: 2 <- 0 <-> 1
        # handle the bidirected edge from 0 to 1
        self.G.remove_edge(0, 1, "directed")
        self.G.add_edge(0, 1, "bidirected")

    def test_size(self):
        G = self.G

        # size stores all edges
        assert G.number_of_edges() == 2
        assert G.number_of_edges(edge_type="directed") == 1
        assert G.number_of_edges(edge_type="bidirected") == 1

    def test_str_unnamed(self):
        G = self.Graph()
        G.add_edges_from([(1, 2), (2, 3)], G.directed_edge_name)
        G.add_edge(1, 3, G.bidirected_edge_name)
        assert str(G) == f"{type(G).__name__} with 3 nodes and 3 edges and 4 edge types"

    def test_str_named(self):
        G = self.Graph(name="foo")
        G.add_edges_from([(1, 2), (2, 3)], G.directed_edge_name)
        G.add_edge(1, 3, G.bidirected_edge_name)
        assert str(G) == f"{type(G).__name__} named 'foo' with 3 nodes and 3 edges and 4 edge types"

    def test_neighbors(self):
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.G.copy()

        # also setup a PAG with uncertain edges
        G.add_edge(1, 4, "circle")
        G.add_edge(4, 1, "circle")

        assert set(G.neighbors(2)) == {0}
        assert set(G.neighbors(0)) == {2, 1}
        assert set(G.neighbors(1)) == {0, 4}
        assert set(G.neighbors(4)) == {1}

    def test_wrong_construction(self):
        # PAGs only allow one type of edge between any two nodes
        edge_list = [
            ("x4", "x1"),
            ("x2", "x5"),
        ]
        latent_edge_list = [("x1", "x2"), ("x4", "x5"), ("x4", "x1")]
        with pytest.raises(
            RuntimeError, match="There is already an existing edge between x4 and x1"
        ):
            PAG(edge_list, incoming_circle_edges=latent_edge_list)

    def test_add_circle_edge(self):
        G = self.G.copy()
        assert not G.has_edge(1, 3, G.directed_edge_name)

        G.add_edge(1, 3, G.circle_edge_name)
        G.add_edge(3, 1, G.circle_edge_name)
        assert not G.has_edge(1, 3, G.directed_edge_name)
        assert G.has_edge(1, 3, G.circle_edge_name)

    def test_adding_edge_errors(self):
        """Test that adding edges in PAG result in certain errors."""
        # 2 <- 0 <-> 1 o-o 4
        G = self.G.copy()
        G.add_edge(1, 4, "circle")
        G.add_edge(4, 1, "circle")

        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_edge(0, 2, G.circle_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_edge(0, 1, G.circle_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_edges_from([(0, 1)], G.circle_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 1 and 4"):
            G.add_edge(1, 4, G.directed_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_edges_from([(0, 1)], G.directed_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_edge(0, 2, G.bidirected_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_edges_from([(0, 2)], G.bidirected_edge_name)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 1 and 4"):
            G.add_edges_from([(1, 4)], G.bidirected_edge_name)
        with pytest.raises(RuntimeError, match="There is an existing 0 -> 2"):
            # adding an edge from 2 -> 0, will result in an error
            G.add_edge(2, 0, G.directed_edge_name)

        # adding a single circle edge is fine
        G.add_edge(2, 0, G.circle_edge_name)

    def test_remove_circle_edge(self):
        G = self.G.copy()
        G.add_edge(1, 4, "circle")
        G.add_edge(4, 1, "circle")

        assert G.has_edge(1, 4, G.circle_edge_name)
        G.remove_edge(1, 4, G.circle_edge_name)
        assert not G.has_edge(1, 4, G.circle_edge_name)

    def test_orient_circle_edge(self):
        G = self.G.copy()
        G.add_edge(1, 4, "circle")
        G.add_edge(4, 1, "circle")

        assert G.has_edge(1, 4, G.circle_edge_name)
        assert not G.has_edge(1, 4, G.directed_edge_name)
        print(G.edges())
        G.orient_uncertain_edge(1, 4)
        print(G.edges())
        assert G.has_edge(1, 4, G.directed_edge_name)
        assert not G.has_edge(1, 4, G.circle_edge_name)

        assert G.has_edge(1, 4, G.directed_edge_name)
        assert not G.has_edge(1, 4, G.circle_edge_name)

    def test_children_and_parents(self):
        """Test working with children and parents."""
        # 2 <- 0 <-> 1 o-o 4
        G = self.G.copy()
        G.add_edge(1, 4, "circle")
        G.add_edge(4, 1, "circle")

        # basic parent/children semantics
        assert [2] == list(G.children(0))
        assert [0] == list(G.parents(2))
        assert [] == list(G.parents(0))
        assert [] == list(G.children(1))
        assert [] == list(G.parents(1))
        assert [] == list(G.parents(4))
        assert [] == list(G.children(4))

        # o-o edges do constitute possible parent/children
        assert [4] == list(G.possible_children(1)) == list(G.possible_parents(1))
        assert [1] == list(G.possible_children(4)) == list(G.possible_parents(4))

        # when the parental relationship between 2 and 0
        # is made uncertain, the parents/children sets reflect
        G.add_edge(2, 0, G.circle_edge_name)
        assert [] == list(G.children(0))
        assert [] == list(G.parents(2))

        # 2 and 0 now have possible children/parents relationship
        assert [0] == list(G.possible_parents(2))
        assert [2] == list(G.possible_children(0))

    # m-separation is not needed to be tested on PAGs
    def test_m_separation(self):
        pass

    # TODO: make work
    @pytest.mark.skip(reason="Need to implement")
    def test_definite_m_separation(self):
        G = self.G.copy()

        # 2 <- 0 <-> 1 o-o 4
        assert not pywhy_nx.m_separated(G, {0}, {4}, set())
        assert not pywhy_nx.m_separated(G, {0}, {4}, {1})

        # check various cases
        G.add_edge(4, 3, G.directed_edge_name)
        assert not pywhy_nx.m_separated(G, {3}, {1}, set())
        assert pywhy_nx.m_separated(G, {3}, {1}, {4})

        # check what happens in the other direction
        G.remove_edge(4, 3, G.directed_edge_name)
        G.add_edge(3, 4, G.directed_edge_name)
        assert not pywhy_nx.m_separated(G, {3}, {1}, set())
        assert not pywhy_nx.m_separated(G, {3}, {1}, {4})
