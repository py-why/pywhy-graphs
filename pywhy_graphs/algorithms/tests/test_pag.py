from itertools import permutations

import pytest

import pywhy_graphs
from pywhy_graphs import PAG
from pywhy_graphs.algorithms import (
    discriminating_path,
    pds,
    pds_path,
    possible_ancestors,
    possible_descendants,
    uncovered_circle_path,
    uncovered_pd_path,
)


@pytest.fixture(scope="module")
def mixed_edge_path_graph():
    """Create a mixed-edge path graph.

    Adds some extra pathways to test.
    """
    # create a path of
    # a -> b o-> c o-o d <-> e
    # with b <- x -> c
    # and d -> f o-> e
    directed_edges = [
        ("a", "b"),
        ("x", "b"),
        ("x", "c"),
        ("b", "c"),
        ("d", "f"),
        ("f", "e"),
    ]
    bidirected_edges = [("d", "e")]
    circle_edges = [
        ("c", "b"),
        ("c", "d"),
        ("d", "c"),
        ("e", "f"),
    ]
    G = pywhy_graphs.PAG(
        incoming_directed_edges=directed_edges,
        incoming_bidirected_edges=bidirected_edges,
        incoming_circle_edges=circle_edges,
    )
    return G


@pytest.fixture(scope="module")
def pds_graph():
    """Figure 6.17 in causation, prediction and search.

    Uses Figure 15, 16, 17 and 18 in "Discovery Algorithms without
    Causal Sufficiency" in :footcite:`Spirtes1993`.

    References
    ----------
    .. footbibliography::
    """
    edge_list = [
        ("D", "A"),
        ("B", "E"),
        ("H", "D"),
        ("F", "B"),
    ]
    latent_edge_list = [("A", "B"), ("D", "E")]
    uncertain_edge_list = [
        ("A", "E"),
        ("E", "A"),
        ("E", "B"),
        ("B", "F"),
        ("F", "C"),
        ("C", "F"),
        ("C", "H"),
        ("H", "C"),
        ("D", "H"),
        ("A", "D"),
    ]
    G = pywhy_graphs.PAG(
        edge_list,
        incoming_bidirected_edges=latent_edge_list,
        incoming_circle_edges=uncertain_edge_list,
    )
    return G


def test_possible_descendants(mixed_edge_path_graph):
    G = mixed_edge_path_graph
    possible_anc = possible_descendants(G, "a")
    assert {"a", "b", "c", "d", "f", "e"} == possible_anc

    possible_anc = possible_descendants(G, "e")
    assert possible_anc == set("e")

    possible_anc = possible_descendants(G, "d")
    assert possible_anc == {"d", "f", "c", "e"}


def test_possible_ancestors(mixed_edge_path_graph):
    G = mixed_edge_path_graph

    possible_anc = possible_ancestors(G, "e")
    assert {"a", "b", "c", "x", "d", "f", "e"} == possible_anc

    possible_anc = possible_ancestors(G, "c")
    assert {"a", "b", "x", "d", "c"} == possible_anc


def test_discriminating_path():
    """Test the output of a discriminating path.

    We look at a graph presented in [1] Figure 2.

    References
    ----------
    [1] Colombo, Diego, et al. "Learning high-dimensional directed acyclic
    graphs with latent and selection variables." The Annals of Statistics
    (2012): 294-321.
    """
    # this is Figure 2's PAG after orienting colliders, there should be no
    # discriminating path
    edges = [
        ("x4", "x1"),
        ("x4", "x6"),
        ("x2", "x5"),
        ("x2", "x6"),
        ("x5", "x6"),
        ("x3", "x4"),
        ("x3", "x2"),
        ("x3", "x6"),
    ]
    bidirected_edges = [("x1", "x2"), ("x4", "x5")]
    circle_edges = [("x4", "x3"), ("x2", "x3"), ("x6", "x2"), ("x6", "x5"), ("x6", "x4")]
    pag = pywhy_graphs.PAG(
        edges, incoming_bidirected_edges=bidirected_edges, incoming_circle_edges=circle_edges
    )

    for u in pag.nodes:
        for (a, c) in permutations(pag.neighbors(u), 2):
            found_discriminating_path, _, _ = discriminating_path(pag, u, a, c, max_path_length=100)
            if (c, u, a) == ("x6", "x3", "x2"):
                assert found_discriminating_path
            else:
                assert not found_discriminating_path

    # by making x5 <- x2 into x5 <-> x2, we will have another discriminating path
    pag.remove_edge("x2", "x5", pag.directed_edge_name)
    pag.add_edge("x5", "x2", pag.bidirected_edge_name)
    for u in pag.nodes:
        for (a, c) in permutations(pag.neighbors(u), 2):
            found_discriminating_path, _, _ = discriminating_path(pag, u, a, c, max_path_length=100)
            if (c, u, a) in (("x6", "x5", "x2"), ("x6", "x3", "x2")):
                assert found_discriminating_path
            else:
                assert not found_discriminating_path

    edges = [
        ("x4", "x1"),
        ("x4", "x6"),
        ("x2", "x5"),
        ("x2", "x6"),
        ("x5", "x6"),
        ("x3", "x4"),
        ("x3", "x2"),
        ("x3", "x6"),
    ]
    bidirected_edges = [("x1", "x2"), ("x4", "x5")]
    circle_edges = [("x4", "x3"), ("x2", "x3"), ("x6", "x4"), ("x6", "x5"), ("x6", "x3")]
    pag = pywhy_graphs.PAG(
        edges, incoming_bidirected_edges=bidirected_edges, incoming_circle_edges=circle_edges
    )
    found_discriminating_path, _, _ = discriminating_path(
        pag, "x3", "x2", "x6", max_path_length=100
    )
    assert found_discriminating_path


def test_uncovered_circle_path():
    # Construct an uncovered circle path A o-o B o-o C
    G = pywhy_graphs.PAG()
    G.add_edge("A", "B", G.circle_edge_name)
    G.add_edge("B", "A", G.circle_edge_name)
    G.add_edge("B", "C", G.circle_edge_name)
    G.add_edge("C", "B", G.circle_edge_name)
    uncov_circle_path, found_uncovered_circle_path = uncovered_circle_path(G, "B", "C", 10, "A")

    assert found_uncovered_circle_path
    assert uncov_circle_path == ["A", "B", "C"]

    # Construct a non-circle path A o-o u o-o B o-> C
    G = pywhy_graphs.PAG()
    G.add_edge("A", "u", G.circle_edge_name)
    G.add_edge("u", "A", G.circle_edge_name)
    G.add_edge("B", "u", G.circle_edge_name)
    G.add_edge("u", "B", G.circle_edge_name)
    G.add_edge("B", "C", G.directed_edge_name)
    G.add_edge("C", "B", G.circle_edge_name)
    uncov_circle_path, found_uncovered_circle_path = uncovered_circle_path(G, "u", "C", 10, "A")

    assert not found_uncovered_circle_path

    # Construct a potentially directed path
    G = pywhy_graphs.PAG()
    G.add_edge("A", "C", G.directed_edge_name)
    G.add_edge("C", "A", G.circle_edge_name)
    G.add_edges_from(
        [("A", "u"), ("u", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
    )
    G.add_edge("y", "x", G.circle_edge_name)

    # create a pd path from A to C through v
    G.add_edges_from(
        [("A", "v"), ("v", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
    )
    # with the bidirected edge, v,x,y is a shielded triple
    G.add_edge("v", "y", G.bidirected_edge_name)

    # check that this is asserted as not a circle path
    _, found_uncovered_circle_path = uncovered_circle_path(G, "u", "C", 100, "A")
    assert not found_uncovered_circle_path


def test_uncovered_pd_path():
    """Test basic uncovered partially directed path."""
    # If A o-> C and there is an undirected pd path
    # from A to C through u, where u and C are not adjacent
    # then orient A o-> C as A -> C
    G = pywhy_graphs.PAG()

    # create an uncovered pd path from A to C through u
    G.add_edge("A", "C", G.directed_edge_name)
    G.add_edge("C", "A", G.circle_edge_name)
    G.add_edges_from(
        [("A", "u"), ("u", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
    )
    G.add_edge("y", "x", G.circle_edge_name)

    # create a pd path from A to C through v
    G.add_edges_from(
        [("A", "v"), ("v", "x"), ("x", "y"), ("y", "z"), ("z", "C")], G.directed_edge_name
    )
    # with the bidirected edge, v,x,y is a shielded triple
    G.add_edge("v", "y", G.bidirected_edge_name)

    # get the uncovered pd paths
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", 100, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # the shielded triple should not result in an uncovered pd path
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "v", "C", 100, "A")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []

    # when there is a circle edge it should still work
    G.add_edge("C", "z", G.circle_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", 100, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # check errors for running uncovered pd path
    with pytest.raises(RuntimeError, match="Both first and second"):
        uncovered_pd_path(G, "u", "C", 100, "A", "x")
    with pytest.raises(RuntimeError, match="Some nodes are not in"):
        uncovered_pd_path(G, "u", "C", 100, "wrong")


def test_uncovered_pd_path_intersecting():
    """Test basic uncovered partially directed path with intersecting paths."""
    G = pywhy_graphs.PAG()

    # make A o-> C
    G.add_edge("A", "C", G.directed_edge_name)
    G.add_edge("C", "A", G.circle_edge_name)
    # create an uncovered pd path from A to u that ends at C
    G.add_edges_from(
        [("A", "x"), ("x", "y"), ("y", "z"), ("z", "u"), ("u", "C")], G.directed_edge_name
    )
    G.add_edge("y", "x", G.circle_edge_name)

    # create an uncovered pd path from A to v so now C is a collider for <u, C, v>
    G.add_edges_from([("z", "v"), ("v", "C")], G.directed_edge_name)
    G_copy = G.copy()

    # get the uncovered pd paths
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", 100, second_node="x")
    assert found_uncovered_pd_path
    assert uncov_pd_path in (["A", "x", "y", "z", "u", "C"], ["A", "x", "y", "z", "v", "C"])

    # when we make the <A, x, y> triple shielded, it is no longer an uncovered path
    G.add_edge("A", "y", G.directed_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", 100, second_node="x")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []

    # For the second test, let's add another uncovered path
    G = G_copy.copy()
    G.add_edges_from([("A", "w"), ("w", "y")], G.directed_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", 100, second_node="w")
    assert found_uncovered_pd_path
    assert uncov_pd_path in (["A", "w", "y", "z", "u", "C"], ["A", "w", "y", "z", "v", "C"])

    # For the third test, the path through x is not an uncovered pd path, but the
    # path through 'y' is
    G = G_copy.copy()
    G.add_edge("A", "y", G.directed_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", 100, second_node="x")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []

    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", 100, second_node="y")
    assert found_uncovered_pd_path
    assert uncov_pd_path in (["A", "y", "z", "u", "C"], ["A", "y", "z", "v", "C"])


def test_possibly_d_separated(pds_graph):
    """Test possibly d-separated set construction.

    Uses Figure 15, 16, 17 and 18 in "Discovery Algorithms without
    Causal Sufficiency" in :footcite:`Spirtes1993`.

    References
    ----------
    .. footbibliography::
    """
    G = pds_graph

    a_pdsep = pds(G, "A", "E")
    e_pdsep = pds(G, "E", "A")

    assert a_pdsep == {"B", "F", "D"}
    assert e_pdsep == {"B", "D", "H"}


def test_pds_path(pds_graph: PAG):
    G = pds_graph

    a_pdspath = pds_path(G, "A", "E")
    e_pdspath = pds_path(G, "E", "A")
    a_pdsep = pds(G, "A", "E")
    e_pdsep = pds(G, "E", "A")

    # the original graph is fully biconnected, so
    # pdspath is equivalent to pds
    assert a_pdsep == a_pdspath
    assert e_pdsep == e_pdspath

    # adding an edge between A and E will not change
    # the earlier. Moreover, if we add a different node
    # that is not biconnected, it will not fall in the
    # pds path
    G.add_edge("A", "E", G.circle_edge_name)
    G.add_edge("E", "A", G.circle_edge_name)
    G.add_edge("x", "E", G.circle_edge_name)
    G.add_edge("E", "x", G.circle_edge_name)
    a_pdspath = pds_path(G, "A", "E")
    e_pdspath = pds_path(G, "E", "A")

    assert a_pdsep == a_pdspath
    assert e_pdsep == e_pdspath

    # since the PDS set does not rely on the second
    # node, the PDS(x, E) is the empty set, while
    # PDS(E, x) comprises now of {B, D, H} and {A}
    # because now A is not the end set
    xe_pdsep = pds(G, "x", "E")
    ex_pdsep = pds(G, "E", "x")
    xe_pdspath = pds_path(G, "x", "E")
    ex_pdspath = pds_path(G, "E", "x")

    assert xe_pdspath == set()
    assert ex_pdspath == set()
    assert xe_pdsep == set()
    assert ex_pdsep == {"A", "B", "D", "H"}
