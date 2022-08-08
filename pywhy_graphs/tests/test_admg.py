from pywhy_graphs import ADMG


class TestADMG(TestDAG, TestExportGraph):
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

    def test_hash(self):
        """Test hashing a causal graph."""
        G = self.G
        current_hash = hash(G)
        assert G._current_hash is None

        G.add_bidirected_edge("1", "2")
        new_hash = hash(G)
        assert current_hash != new_hash

        G.remove_bidirected_edge("1", "2")
        assert current_hash == hash(G)

    def test_full_graph(self):
        """Test computing a full graph from causal graph."""
        G = self.G
        # the current hash should match after computing full graphs
        current_hash = hash(G)
        G.compute_full_graph()
        assert current_hash == G._current_hash
        G.compute_full_graph()
        assert current_hash == G._current_hash

        # after adding a new edge, the hash should change and
        # be different
        G.add_bidirected_edge("1", "2")
        new_hash = hash(G)
        assert new_hash != G._current_hash

        # once the hash is computed, it should be the same again
        G.compute_full_graph()
        assert new_hash == G._current_hash

        # removing the bidirected edge should result in the same
        # hash again
        G.remove_bidirected_edge("1", "2")
        assert current_hash != G._current_hash
        G.compute_full_graph()
        assert current_hash == G._current_hash

        # different orders of edges shouldn't matter
        G_copy = G.copy()
        G.add_bidirected_edge("1", "2")
        G.add_bidirected_edge("2", "3")
        G_hash = hash(G)
        G_copy.add_bidirected_edge("2", "3")
        G_copy.add_bidirected_edge("1", "2")
        copy_hash = hash(G_copy)
        assert G_hash == copy_hash

    def test_bidirected_edge(self):
        """Test bidirected edge functions."""
        # add bidirected edge to an isolated node
        G = self.G
        G.add_bidirected_edge(1, 5)
        assert G.has_bidirected_edge(1, 5)
        assert G.has_bidirected_edge(5, 1)
        G.remove_bidirected_edge(1, 5, remove_isolate=False)
        assert 5 in G
        assert nx.is_isolate(G, 5)
        assert not G.has_bidirected_edge(1, 5)
        assert not G.has_bidirected_edge(5, 1)

        G.add_bidirected_edge(1, 5)
        G.remove_bidirected_edge(1, 5)
        print(G.nodes)
        assert 5 not in G

    def test_m_separation(self):
        G = self.G.copy()
        # add collider on 0
        G.add_edge(3, 0)

        # normal d-separation statements should hold
        assert not d_separated(G, 1, 2, set())
        assert not d_separated(G, 1, 2)
        assert d_separated(G, 1, 2, 0)

        # when we add an edge from 0 -> 1
        # there is no d-separation statement
        assert not d_separated(G, 3, 1, set())
        assert not d_separated(G, 3, 1, 0)

        # test collider works on bidirected edge
        # 1 <-> 0
        G.remove_edge(0, 1)
        assert d_separated(G, 3, 1, set())
        assert not d_separated(G, 3, 1, 0)

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
        G.add_bidirected_edge(2, 3)
        assert [] == list(G.parents(3))
        assert [] == list(G.children(3))

    def test_size(self):
        G = self.G

        # size stores all edges
        assert G.size() == 3
        assert G.number_of_edges() == 2
        assert G.number_of_bidirected_edges() == 1

    def test_c_components(self):
        """Test working with c-components in causal graph."""
        pass