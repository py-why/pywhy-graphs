from itertools import chain, combinations

import networkx as nx

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def test_acyclification():
    """Test acyclification procedure as specified in :footcite:`Mooij2020cyclic`.

    Tests the graphs as presented in Figure 2.
    """
    # cycle with x2 -> x4 -> x6 -> x5 -> x3 -> x4
    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x4", "x6"),  # start of cycle
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),  # end of cycle
            ("x6", "x7"),
        ]
    )
    bidirected_edges = nx.Graph([("x1", "x3")])
    G = pywhy_nx.MixedEdgeGraph([directed_edges, bidirected_edges], ["directed", "bidirected"])
    acyclic_G = pywhy_graphs.acyclification(G)

    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x6", "x7"),
            ("x2", "x3"),
            ("x2", "x5"),
            ("x2", "x4"),
            ("x2", "x6"),
        ]
    )
    bidirected_edges = nx.Graph(
        [
            ("x1", "x3"),
            ("x4", "x6"),
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),
            ("x4", "x5"),
            ("x3", "x6"),
            ("x1", "x3"),
            ("x1", "x5"),
            ("x1", "x4"),
            ("x1", "x6"),
        ]
    )
    expected_G = pywhy_nx.MixedEdgeGraph(
        [directed_edges, bidirected_edges], ["directed", "bidirected"]
    )

    for edge_type, graph in acyclic_G.get_graphs().items():
        expected_graph = expected_G.get_graphs(edge_type)
        assert nx.is_isomorphic(graph, expected_graph)


def test_sigma_separated():
    """Test sigma-separated procedure.

    Note: sigma-separation is impossible within a cycle (i.e. same
    strongly connected component).
    """
    # create a circular graph from 0 -> ... -> 4 -> 0
    cyclic_G = nx.circulant_graph(5, offsets=[1], create_using=nx.DiGraph)
    cyclic_G = pywhy_nx.MixedEdgeGraph(graphs=[cyclic_G], edge_types=["directed"])
    cyclic_G.add_edge_type(nx.Graph(), edge_type="bidirected")

    for u, v in combinations(cyclic_G.nodes, 2):
        other_nodes = set(cyclic_G.nodes)
        other_nodes.remove(u)
        other_nodes.remove(v)
        for z in powerset(other_nodes):
            assert not pywhy_graphs.sigma_separated(cyclic_G, {u}, {v}, set(z))

    # on the other hand, if there is a descendant of a node within the cycle,
    # we can sigma-separate
    cyclic_G.add_edge(3, "x", edge_type="directed")
    other_nodes = set(cyclic_G.nodes)
    other_nodes.remove(3)
    other_nodes.remove("x")
    for u in other_nodes:
        assert pywhy_graphs.sigma_separated(cyclic_G, {u}, {"x"}, {3})
