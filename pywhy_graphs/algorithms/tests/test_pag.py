from itertools import permutations

import pytest

import pywhy_graphs
from pywhy_graphs import PAG
from pywhy_graphs.algorithms import (
    discriminating_path,
    is_definite_noncollider,
    pds,
    pds_path,
    pds_t,
    pds_t_path,
    possible_ancestors,
    possible_descendants,
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


@pytest.fixture(scope="module")
def pdst_graph():
    """Extension of the Figure 6.17 in "causation, prediction and search" for time-series.

    Creates the relevant graph, but now with time lag points.

    References
    ----------
    .. footbibliography::
    """
    edge_list = [
        (("D", -1), ("A", 0)),
        (("B", -1), ("E", 0)),
        (("H", -1), ("D", 0)),
        (("F", -1), ("B", 0)),
    ]
    latent_edge_list = [(("A", 0), ("B", -1)), (("D", 0), ("E", 0))]
    uncertain_edge_list = [
        (("A", 0), ("E", 0)),
        (("E", 0), ("A", 0)),
        (("E", 0), ("B", -1)),
        (("B", -1), ("F", -1)),
        (("F", -1), ("C", -1)),
        (("C", -1), ("F", -1)),
        (("C", -1), ("H", -1)),
        (("H", -1), ("C", -1)),
        (("D", 0), ("H", -1)),
        (("A", 0), ("D", 0)),
    ]
    G = pywhy_graphs.StationaryTimeSeriesPAG(
        edge_list,
        incoming_bidirected_edges=latent_edge_list,
        incoming_circle_edges=uncertain_edge_list,
        max_lag=2,
        stationary=False,
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
    # discriminating path, except one
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
        for a, c in permutations(pag.neighbors(u), 2):
            found_discriminating_path, disc_path, _ = discriminating_path(
                pag, u, a, c, max_path_length=100
            )
            if (c, u, a) == ("x6", "x3", "x2"):
                assert found_discriminating_path
                assert list(disc_path) == ["x1", "x2", "x3", "x6"]
            else:
                assert not found_discriminating_path

    # by making x5 <- x2 into x5 <-> x2, we will have another discriminating path
    pag.remove_edge("x2", "x5", pag.directed_edge_name)
    pag.add_edge("x5", "x2", pag.bidirected_edge_name)
    for u in pag.nodes:
        for a, c in permutations(pag.neighbors(u), 2):
            found_discriminating_path, disc_path, _ = discriminating_path(
                pag, u, a, c, max_path_length=100
            )
            if (c, u, a) == ("x6", "x5", "x2"):
                assert found_discriminating_path
                assert list(disc_path) == ["x1", "x2", "x5", "x6"]
            elif (c, u, a) == ("x6", "x3", "x2"):
                assert found_discriminating_path
                assert list(disc_path) == ["x1", "x2", "x3", "x6"]
            else:
                assert not found_discriminating_path


@pytest.mark.parametrize("xp_xb", ["circle", "bidirected", None])
@pytest.mark.parametrize("xj_xb", ["circle", "bidirected", None])
def test_discriminating_path_longer(xj_xb, xp_xb):
    """Test discriminating path for Figure 4 from the supplemental of [1].

    The endpoints indicated by circle endpoints can be anything:

    - xj <-* xb
    - xp <-* xb

    References
    ----------
    [1] Colombo, Diego, et al. "Learning high-dimensional directed acyclic
    graphs with latent and selection variables." The Annals of Statistics
    (2012): 294-321.
    """
    # test against longer discriminating path with Figure 4
    edges = [
        ("xk", "xp"),
        ("xj", "xp"),
        ("xb", "xp"),
        ("xb", "xj"),
    ]
    bidirected_edges = [("xl", "xk"), ("xk", "xj")]
    pag = pywhy_graphs.PAG(edges, incoming_bidirected_edges=bidirected_edges)
    if xp_xb == "bidirected":
        pag.remove_edge("xb", "xp", pag.directed_edge_name)
        pag.add_edge("xp", "xb", xp_xb)
    elif xp_xb is not None:
        pag.add_edge("xp", "xb", xp_xb)

    if xj_xb == "bidirected":
        pag.remove_edge("xb", "xj", pag.directed_edge_name)
        pag.add_edge("xj", "xb", xj_xb)
    elif xj_xb is not None:
        pag.add_edge("xj", "xb", xj_xb)

    found_discriminating_path, disc_path, _ = discriminating_path(
        pag, "xb", "xj", "xp", max_path_length=10
    )
    assert found_discriminating_path
    assert list(disc_path) == ["xl", "xk", "xj", "xb", "xp"]


def test_restricted_discriminating_path():
    """Test computing discriminating path with path length restrictions."""
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
    found_discriminating_path, disc_path, _ = discriminating_path(
        pag, "x3", "x2", "x6", max_path_length=10
    )
    assert found_discriminating_path
    assert list(disc_path) == ["x1", "x2", "x3", "x6"]

    found_discriminating_path, disc_path, _ = discriminating_path(
        pag, "x3", "x2", "x6", max_path_length=0
    )
    assert not found_discriminating_path


def test_uncovered_pd_path_circle_path_only():
    # Construct an uncovered circle path A o-o B o-o C
    G = pywhy_graphs.PAG()
    G.add_edge("A", "B", G.circle_edge_name)
    G.add_edge("B", "A", G.circle_edge_name)
    G.add_edge("B", "C", G.circle_edge_name)
    G.add_edge("C", "B", G.circle_edge_name)
    uncov_circle_path, found_uncovered_circle_path = uncovered_pd_path(
        G, "B", "C", 10, "A", force_circle=True
    )

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
    uncov_circle_path, found_uncovered_circle_path = uncovered_pd_path(
        G, "u", "C", 10, "A", force_circle=True
    )

    assert not found_uncovered_circle_path

    # Construct A o-o C, forbid C as the first node from A, and check that
    # no circle path was found
    G = pywhy_graphs.PAG()
    G.add_edge("A", "C", G.circle_edge_name)
    G.add_edge("C", "A", G.circle_edge_name)
    uncov_circle_path, found_uncovered_circle_path = uncovered_pd_path(
        G, "A", "C", 10, force_circle=True, forbid_node="C"
    )

    assert not found_uncovered_circle_path

    # Construct a potentially directed path that is not a circle path, and check that it
    # is not detected if force_circle=True
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
    _, found_uncovered_circle_path = uncovered_pd_path(G, "u", "C", 100, "A", force_circle=True)
    assert not found_uncovered_circle_path


@pytest.mark.parametrize("max_path_length", [10, None])
def test_uncovered_pd_path(max_path_length):
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
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", max_path_length, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # the shielded triple should not result in an uncovered pd path
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "v", "C", max_path_length, "A")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []

    # when there is a circle edge it should still work
    G.add_edge("C", "z", G.circle_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", max_path_length, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # Check that a circle path A o-o u o-o C is identified as an uncovered pd path
    G = pywhy_graphs.PAG()
    G.add_edge("A", "u", G.circle_edge_name)
    G.add_edge("u", "A", G.circle_edge_name)
    G.add_edge("u", "C", G.circle_edge_name)
    G.add_edge("C", "u", G.circle_edge_name)
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "A", "C", max_path_length)
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "C"]

    # check errors for running uncovered pd path
    with pytest.raises(RuntimeError, match="Both first and second"):
        uncovered_pd_path(G, "u", "C", max_path_length, "A", "x")
    with pytest.raises(RuntimeError, match="Some nodes are not in"):
        uncovered_pd_path(G, "u", "C", max_path_length, "wrong")


def test_uncovered_pd_path_restricted():
    """Test restriction of max path length in uncovered pd path."""

    # use the same setup as in test_uncovered_pd_path
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

    # get the uncovered pd paths
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", None, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # if we limit the path length, then we won't find a uncovered pd path
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", 3, "A")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []


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


def test_pds_unconnected(pds_graph: PAG):
    """Test PDS in edge case where nodes are unconnected."""
    G = pds_graph
    G.add_node("N")

    pds_set = pds(G, "N", "E")
    assert pds_set == set()


def test_definite_non_collider():
    """Test non-collider definite status check."""
    G = PAG()
    G.add_nodes_from(["x", "y", "z"])

    # first test x *-* y -> z
    G_copy = G.copy()
    G_copy.add_edge("y", "z", G_copy.directed_edge_name)
    G_copy.add_edge("x", "y", G_copy.directed_edge_name)

    assert is_definite_noncollider(G_copy, "x", "y", "z")

    # even if the triplet is shielded, it is a definite collider
    G_copy.add_edge("x", "z", G_copy.bidirected_edge_name)
    assert is_definite_noncollider(G_copy, "x", "y", "z")

    # x <-> y <-> z
    G_copy.remove_edge("y", "z", G_copy.directed_edge_name)
    G_copy.add_edge("y", "z", G_copy.bidirected_edge_name)
    assert not is_definite_noncollider(G_copy, "x", "y", "z")

    # second test x <- y *-* z
    G_copy = G.copy()
    G_copy.add_edge("y", "x", G_copy.directed_edge_name)
    G_copy.add_edge("z", "y", G_copy.bidirected_edge_name)
    assert is_definite_noncollider(G_copy, "x", "y", "z")

    # even if the triplet is shielded, it is a definite collider
    G_copy.add_edge("x", "z", G_copy.bidirected_edge_name)
    assert is_definite_noncollider(G_copy, "x", "y", "z")

    G_copy.remove_edge("y", "x", G_copy.directed_edge_name)
    G_copy.add_edge("y", "x", G_copy.bidirected_edge_name)
    assert not is_definite_noncollider(G_copy, "x", "y", "z")

    # third test x o-o y o-o z, where x and z are unshielded
    G_copy = G.copy()
    G_copy.add_edge("y", "x", G_copy.circle_edge_name)
    G_copy.add_edge("x", "y", G_copy.circle_edge_name)
    G_copy.add_edge("z", "y", G_copy.circle_edge_name)
    G_copy.add_edge("y", "z", G_copy.circle_edge_name)

    assert is_definite_noncollider(G_copy, "x", "y", "z")

    # even if the triplet is shielded, it is a definite collider
    G_copy.add_edge("x", "z", G_copy.directed_edge_name)
    assert not is_definite_noncollider(G_copy, "x", "y", "z")


def test_pdst(pdst_graph):
    G = pdst_graph

    a_pdspath = pds_t_path(G, ("A", 0), ("E", 0))
    e_pdspath = pds_t_path(G, ("E", 0), ("A", 0))
    a_pdsep = pds_t(G, ("A", 0), ("E", 0))
    e_pdsep = pds_t(G, ("E", 0), ("A", 0))

    # the original graph is fully biconnected, so
    # pdspath is equivalent to pds
    assert a_pdsep == a_pdspath
    assert e_pdsep == e_pdspath

    # If we add a different node that is not biconnected,
    # it will not fall in the pds path
    G.add_edge(("x", -1), ("E", 0), G.circle_edge_name)
    G.add_edge(("E", 0), ("x", -1), G.circle_edge_name)
    a_pdspath = pds_t_path(G, ("A", 0), ("E", 0))
    e_pdspath = pds_t_path(G, ("E", 0), ("A", 0))

    assert a_pdsep == a_pdspath
    assert e_pdsep == e_pdspath

    # since the PDS set does not rely on the second
    # node, the PDS(x, E) is the empty set, while
    # PDS(E, x) comprises now of {B, D, H} and {A}
    # because now A is not the end set
    xe_pdsep = pds_t(G, ("x", -1), ("E", 0))
    ex_pdsep = pds_t(G, ("E", 0), ("x", -1))
    xe_pdspath = pds_t_path(G, ("x", -1), ("E", 0))
    ex_pdspath = pds_t_path(G, ("E", 0), ("x", -1))

    assert xe_pdspath == set()
    assert ex_pdspath == set()
    assert xe_pdsep == set()
    assert ex_pdsep == {("A", 0), ("B", -1), ("D", 0), ("H", -1)}

    # now we look at variables beyond the max(lag(node1), lag(node2)),
    # these are not included even if they would be included in the PDS
    G.add_edge(("x", -1), ("y", -2), G.circle_edge_name)
    G.add_edge(("y", -2), ("x", -1), G.circle_edge_name)
    G.add_edge(("y", -2), ("E", 0), G.circle_edge_name)
    G.add_edge(("E", 0), ("y", -2), G.circle_edge_name)

    xe_pdsep = pds(G, ("x", -1), ("E", 0))
    ex_pdsep = pds(G, ("E", 0), ("x", -1))
    assert ("y", -2) in xe_pdsep
    assert ("y", -2) in ex_pdsep

    xe_pdsep_t = pds_t(G, ("x", -1), ("E", 0))
    ex_pdsep_t = pds_t(G, ("E", 0), ("x", -1))
    assert ("y", -2) not in xe_pdsep_t
    assert ("y", -2) not in ex_pdsep_t


def test_pag_to_mag():
    # C o- A o-> D <-o B
    # B o-o A o-o C o-> D

    pag = PAG()
    pag.add_edge("A", "D", pag.directed_edge_name)
    pag.add_edge("A", "C", pag.circle_edge_name)
    pag.add_edge("D", "A", pag.circle_edge_name)
    pag.add_edge("B", "D", pag.directed_edge_name)
    pag.add_edge("C", "D", pag.directed_edge_name)
    pag.add_edge("D", "B", pag.circle_edge_name)
    pag.add_edge("D", "C", pag.circle_edge_name)
    pag.add_edge("C", "A", pag.circle_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("A", "B", pag.circle_edge_name)

    out_mag = pywhy_graphs.pag_to_mag(pag)

    # C <- A -> B -> D or C -> A -> B -> D or C <- A <- B -> D
    # A -> D <- C

    assert (
        ((out_mag.has_edge("A", "B")) or (out_mag.has_edge("B", "A")))
        and ((out_mag.has_edge("A", "C")) or (out_mag.has_edge("C", "A")))
        and (out_mag.has_edge("A", "D"))
        and (out_mag.has_edge("B", "D"))
        and (out_mag.has_edge("C", "D"))
    )

    # D o-> A <-o B
    # D o-o B
    pag = PAG()
    pag.add_edge("A", "B", pag.circle_edge_name)
    pag.add_edge("B", "A", pag.directed_edge_name)
    pag.add_edge("D", "A", pag.directed_edge_name)
    pag.add_edge("A", "D", pag.circle_edge_name)
    pag.add_edge("D", "B", pag.circle_edge_name)
    pag.add_edge("B", "D", pag.circle_edge_name)

    out_mag = pywhy_graphs.pag_to_mag(pag)

    # B -> A <- D
    # D -> B or D <- B

    assert (
        out_mag.has_edge("B", "A")
        and out_mag.has_edge("D", "A")
        and (out_mag.has_edge("D", "B") or out_mag.has_edge("B", "D"))
    )

    # A -> B <- C o-o D
    # D o-o E -> B

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("E", "B", pag.directed_edge_name)
    pag.add_edge("E", "D", pag.circle_edge_name)
    pag.add_edge("C", "D", pag.circle_edge_name)
    pag.add_edge("D", "E", pag.circle_edge_name)
    pag.add_edge("D", "C", pag.circle_edge_name)

    out_mag = pywhy_graphs.pag_to_mag(pag)

    # A -> B <- C <- D or A -> B <- C -> D
    # D <- E -> B or D <- E -> B

    assert (
        out_mag.has_edge("A", "B")
        and out_mag.has_edge("C", "B")
        and out_mag.has_edge("E", "B")
        and (out_mag.has_edge("E", "D") or out_mag.has_edge("D", "E"))
        and (out_mag.has_edge("D", "C") or out_mag.has_edge("C", "D"))
    )


def test_check_pag_definition():
    # D o-o A o-> B <-o C

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("B", "C", pag.circle_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("A", "D", pag.circle_edge_name)
    pag.add_edge("D", "A", pag.circle_edge_name)

    pag_bool = pywhy_graphs.check_pag_definition(pag)

    assert pag_bool is True

    # D <-> A o-> B <-o C
    # B -> D

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("B", "C", pag.circle_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("A", "D", pag.bidirected_edge_name)
    pag.add_edge("B", "D", pag.directed_edge_name)

    pag_bool = pywhy_graphs.check_pag_definition(pag)

    assert pag_bool is False

    # D -> A o-> B <-o C
    # B -> D

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("B", "C", pag.circle_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("D", "A", pag.directed_edge_name)
    pag.add_edge("B", "D", pag.directed_edge_name)

    pag_bool = pywhy_graphs.check_pag_definition(pag)

    assert pag_bool is False


def test_valid_pag():

    pag = PAG()
    pag.add_edge("A", "D", pag.directed_edge_name)
    pag.add_edge("A", "C", pag.circle_edge_name)
    pag.add_edge("D", "A", pag.circle_edge_name)
    pag.add_edge("B", "D", pag.directed_edge_name)
    pag.add_edge("C", "D", pag.directed_edge_name)
    pag.add_edge("D", "B", pag.circle_edge_name)
    pag.add_edge("D", "C", pag.circle_edge_name)
    pag.add_edge("C", "A", pag.circle_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("A", "B", pag.circle_edge_name)

    # C o- A o-> D <-o B
    # B o-o A o-o C o-> D

    assert pywhy_graphs.valid_pag(pag) is True

    pag = PAG()
    pag.add_edge("A", "B", pag.circle_edge_name)
    pag.add_edge("B", "A", pag.directed_edge_name)
    pag.add_edge("D", "A", pag.directed_edge_name)
    pag.add_edge("A", "D", pag.circle_edge_name)
    pag.add_edge("D", "B", pag.circle_edge_name)
    pag.add_edge("B", "D", pag.circle_edge_name)

    # D o-> A <-o B
    # D o-o B
    assert pywhy_graphs.valid_pag(pag) is False

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("E", "B", pag.directed_edge_name)
    pag.add_edge("E", "D", pag.circle_edge_name)
    pag.add_edge("C", "D", pag.circle_edge_name)
    pag.add_edge("D", "E", pag.circle_edge_name)
    pag.add_edge("D", "C", pag.circle_edge_name)

    # A -> B <- C o-o D
    # D o-o E -> B

    assert pywhy_graphs.valid_pag(pag) is False

    pag = PAG()
    pag.add_edge("A", "B", pag.directed_edge_name)
    pag.add_edge("B", "A", pag.circle_edge_name)
    pag.add_edge("C", "B", pag.directed_edge_name)
    pag.add_edge("E", "B", pag.directed_edge_name)
    pag.add_edge("E", "D", pag.circle_edge_name)
    pag.add_edge("C", "D", pag.circle_edge_name)
    pag.add_edge("D", "E", pag.circle_edge_name)
    pag.add_edge("D", "C", pag.circle_edge_name)

    # A o-> B <- C o-o D
    # D o-o E -> B

    assert pywhy_graphs.valid_pag(pag) is True
