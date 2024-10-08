import networkx as nx
import pytest

import pywhy_graphs
from pywhy_graphs import ADMG, PAG
from pywhy_graphs.algorithms import all_vstructures


def test_convert_to_latent_confounder_errors():
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = pywhy_graphs.ADMG(incoming_graph_data)

    with pytest.raises(RuntimeError, match="is not a common cause within the graph"):
        pywhy_graphs.set_nodes_as_latent_confounders(G, [1])


@pytest.mark.parametrize("graph_func", [pywhy_graphs.ADMG, nx.DiGraph])
def test_convert_to_latent_confounder(graph_func):
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = graph_func(incoming_graph_data)

    assert pywhy_graphs.is_node_common_cause(G, 0)
    assert not pywhy_graphs.is_node_common_cause(G, 0, exclude_nodes=set([1]))

    # remove 0 and set a bidirected edge between 1 <--> 2
    # 1 <--> 2 <- 3, so 3 is independent of 1, but everything else is connected
    # the collider should be orientable.
    G = pywhy_graphs.set_nodes_as_latent_confounders(G, [0])

    expected_G = pywhy_graphs.ADMG([(3, 2)], incoming_bidirected_edges=[(1, 2)])
    assert nx.is_isomorphic(G.to_undirected(), expected_G.to_undirected())
    assert expected_G.edges() == G.edges()

    G.add_edge(3, 1, G.bidirected_edge_name)
    assert not pywhy_graphs.is_node_common_cause(G, 3)

    G.remove_edge(3, 1, G.bidirected_edge_name)
    G.add_edge(3, 1, G.directed_edge_name)
    assert pywhy_graphs.is_node_common_cause(G, 3)


def test_inducing_path():
    admg = ADMG()

    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Z", "Y", admg.bidirected_edge_name)
    admg.add_edge("Z", "H", admg.bidirected_edge_name)

    # X -> Y <-> z <-> H

    S = {"Y", "Z"}
    L = set()
    assert pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    admg.add_edge("H", "J", admg.directed_edge_name)

    # X -> Y <-> z <-> H -> J

    S = {"Y", "Z"}
    L = {"H"}

    assert pywhy_graphs.inducing_path(admg, "X", "J", L, S)[0]

    admg.add_edge("K", "J", admg.directed_edge_name)

    # X -> Y <-> z <-> H -> J <- K
    S = {"Y", "Z", "J"}
    L = {"H"}

    assert pywhy_graphs.inducing_path(admg, "X", "K", L, S)[0]  # no directed path exists

    admg.add_edge("J", "K", admg.directed_edge_name)

    # X -> Y <-> z <-> H -> J <-> K

    S = {"Y", "J"}
    L = {"H"}

    assert not pywhy_graphs.inducing_path(admg, "X", "K", L, S)[
        0
    ]  # A collider on the path is not in S

    S = {"Y", "Z"}
    L = set()

    assert not pywhy_graphs.inducing_path(admg, "X", "K", L, S)[
        0
    ]  # A non-collider on the path is not in S


def test_inducing_path_wihtout_LandS():
    admg = ADMG()

    admg.add_edge("X", "Y", admg.directed_edge_name)

    L = set()
    S = set()

    # X -> Y

    assert pywhy_graphs.inducing_path(admg, "X", "Y", L, S)[0]

    admg.add_edge("Y", "X", admg.directed_edge_name)

    # X <-> Y

    assert pywhy_graphs.inducing_path(admg, "X", "Y", L, S)[0]


def test_inducing_path_one_direction():
    admg = ADMG()

    admg.add_edge("A", "B", admg.directed_edge_name)
    admg.add_edge("B", "C", admg.directed_edge_name)
    admg.add_edge("C", "D", admg.directed_edge_name)
    admg.add_edge("B", "C", admg.bidirected_edge_name)

    L = {"C"}
    S = {"B"}

    # A -> B -> C -> D
    # B <-> C

    assert pywhy_graphs.inducing_path(admg, "A", "D", L, S)[0]

    L = set()
    S = {"B"}

    assert not pywhy_graphs.inducing_path(admg, "A", "D", L, S)[0]

    L = {"C"}
    S = set()

    assert pywhy_graphs.inducing_path(admg, "A", "D", L, S)[0]

    admg.add_edge("D", "C", admg.bidirected_edge_name)

    # A -> B -> C -> D
    # B <-> C
    # C <-> D

    L = set()
    S = {"B"}

    assert pywhy_graphs.inducing_path(admg, "A", "D", L, S)[0]

    L = set()
    S = set()

    assert pywhy_graphs.inducing_path(admg, "A", "D", L, S)[0]


def test_inducing_path_corner_cases():
    # X <- Y <-> Z <-> H; Z -> X
    admg = ADMG()
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("Z", "Y", admg.bidirected_edge_name)
    admg.add_edge("Z", "H", admg.bidirected_edge_name)

    # not an inducing path, since Y is not a collider and Y is not part of L
    S = {}
    L = {}
    assert not pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # now an inducing path, since Y is not a collider, but is part of L
    L = {"Y"}
    assert pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # X <-> Y <-> Z <-> H; Z -> X
    admg = ADMG()
    admg.add_edge("Y", "X", admg.bidirected_edge_name)
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("Z", "Y", admg.bidirected_edge_name)
    admg.add_edge("Z", "H", admg.bidirected_edge_name)

    # not an inducing path, since Y is not an ancestor of X, H, or S
    S = {}
    L = {}
    assert not pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # still not an inducing path, since Y is a collider
    L = {"Y"}
    assert not pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # now add an edge Y -> A
    admg.add_edge("Y", "A", admg.directed_edge_name)

    # an inducing path, since Y is a collider and is an ancestor of X, H, or S
    L = {}
    S = {"A"}
    assert pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # an inducing path, since Y is a collider and is an ancestor of X, H, or S
    L = {}
    S = {"Y"}
    assert pywhy_graphs.inducing_path(admg, "X", "H", L, S)[0]

    # X -> Z <- Y, A <- B <- Z
    admg = ADMG()
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Y", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "B", admg.directed_edge_name)
    admg.add_edge("B", "A", admg.directed_edge_name)

    L = {}
    S = {"A"}

    assert pywhy_graphs.inducing_path(admg, "X", "Y", L, S)[0]

    # X -> Z <- Y, A <- B <- Z
    admg = ADMG()
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Y", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "B", admg.directed_edge_name)
    admg.add_edge("B", "A", admg.directed_edge_name)

    L = {"X"}
    S = {"A"}

    assert not pywhy_graphs.inducing_path(admg, "X", "Y", L, S)[0]

    # X -> Z <- Y, A <- B <- Z
    admg = ADMG()
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Y", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "B", admg.directed_edge_name)
    admg.add_edge("B", "A", admg.directed_edge_name)

    L = {}
    S = {"A", "Y"}

    assert not pywhy_graphs.inducing_path(admg, "X", "Y", L, S)[0]


def test_is_collider():
    # Z -> X -> A <- B -> Y; H -> A
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("H", "A", admg.directed_edge_name)
    admg.add_edge("X", "A", admg.directed_edge_name)
    admg.add_edge("B", "A", admg.directed_edge_name)
    admg.add_edge("B", "Y", admg.directed_edge_name)

    L = {"X", "B"}
    S = {"A"}

    assert pywhy_graphs.inducing_path(admg, "Z", "Y", L, S)[0]


def test_has_adc():
    # K -> H -> Z -> X -> Y -> J <- K
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.directed_edge_name)

    assert not pywhy_graphs.has_adc(admg)  # there is no cycle completed by a bidirected edge

    # K -> H -> Z -> X -> Y -> J <-> K
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.bidirected_edge_name)

    assert pywhy_graphs.has_adc(admg)  # there is a bidirected edge from J to K, completing a cycle

    # K -> H -> Z -> X -> Y <- J <-> K
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("J", "Y", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.bidirected_edge_name)

    assert not pywhy_graphs.has_adc(admg)  # Y <- J is not correctly oriented

    # I -> H -> Z -> X -> Y -> J <-> K
    # J -> I
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("Y", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.bidirected_edge_name)

    assert pywhy_graphs.has_adc(admg)  # J <-> K completes an otherwise directed cycle


def test_valid_mag():
    # K -> H -> Z -> X -> Y -> J <- K
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.directed_edge_name)

    S = {"J"}
    L = {}

    assert not pywhy_graphs.valid_mag(
        admg, L, S  # J is in S and is a collider on the path Y -> J <- K
    )

    S = {}

    assert pywhy_graphs.valid_mag(admg, L, S)  # there are no valid inducing paths

    # K -> H -> Z -> X -> Y -> J -> K
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("J", "K", admg.directed_edge_name)

    L = {}

    assert not pywhy_graphs.valid_mag(admg, L, S)  # there is a directed cycle

    # K -> H -> Z -> X -> Y -> J <- K
    # H <-> J
    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.directed_edge_name)
    admg.add_edge("H", "J", admg.bidirected_edge_name)

    assert not pywhy_graphs.valid_mag(admg)  # there is an almost directed cycle

    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.directed_edge_name)
    admg.add_edge("H", "J", admg.bidirected_edge_name)
    admg.add_edge("H", "J", admg.directed_edge_name)

    assert not pywhy_graphs.valid_mag(admg)  # there are two edges between H and J

    admg = ADMG()
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("X", "Y", admg.directed_edge_name)
    admg.add_edge("Y", "J", admg.directed_edge_name)
    admg.add_edge("H", "Z", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)
    admg.add_edge("K", "J", admg.directed_edge_name)
    admg.add_edge("H", "J", admg.undirected_edge_name)

    assert not pywhy_graphs.valid_mag(admg)  # there is an undirected edge between H and J


def test_dag_to_mag():
    # A -> E -> S
    # H -> E , H -> R
    admg = ADMG()
    admg.add_edge("A", "E", admg.directed_edge_name)
    admg.add_edge("E", "S", admg.directed_edge_name)
    admg.add_edge("H", "E", admg.directed_edge_name)
    admg.add_edge("H", "R", admg.directed_edge_name)

    S = {"S"}
    L = {"H"}

    out_mag = pywhy_graphs.dag_to_mag(admg, L, S)
    assert pywhy_graphs.is_maximal(out_mag)
    assert not pywhy_graphs.has_adc(out_mag)
    out_edges = out_mag.edges()
    dir_edges = list(out_edges["directed"])
    assert (
        ("A", "R") in out_edges["directed"]
        and ("E", "R") in out_edges["directed"]
        and len(out_edges["directed"]) == 2
    )
    assert ("A", "E") in out_edges["undirected"]

    out_mag = pywhy_graphs.dag_to_mag(admg)
    dir_edges = list(out_mag.edges()["directed"])

    assert (
        ("A", "E") in dir_edges
        and ("E", "S") in dir_edges
        and ("H", "E") in dir_edges
        and ("H", "R") in dir_edges
    )

    # A -> E -> S <- H
    # H -> E , H -> R,

    admg = ADMG()
    admg.add_edge("A", "E", admg.directed_edge_name)
    admg.add_edge("H", "S", admg.directed_edge_name)
    admg.add_edge("H", "E", admg.directed_edge_name)
    admg.add_edge("H", "R", admg.directed_edge_name)

    S = {"S"}
    L = {"H"}

    out_mag = pywhy_graphs.dag_to_mag(admg, L, S)
    assert pywhy_graphs.is_maximal(out_mag)
    assert not pywhy_graphs.has_adc(out_mag)
    out_edges = out_mag.edges()

    dir_edges = list(out_edges["directed"])
    assert ("A", "E") in out_edges["directed"] and len(out_edges["directed"]) == 1
    assert ("E", "R") in out_edges["bidirected"]

    # P -> S -> L <- G
    # G -> S -> I <- J
    # J -> S

    admg = ADMG()
    admg.add_edge("P", "S", admg.directed_edge_name)
    admg.add_edge("S", "L", admg.directed_edge_name)
    admg.add_edge("G", "S", admg.directed_edge_name)
    admg.add_edge("G", "L", admg.directed_edge_name)
    admg.add_edge("I", "S", admg.directed_edge_name)
    admg.add_edge("J", "I", admg.directed_edge_name)
    admg.add_edge("J", "S", admg.directed_edge_name)

    S = set()
    L = {"J"}

    out_mag = pywhy_graphs.dag_to_mag(admg, L, S)
    assert pywhy_graphs.is_maximal(out_mag)
    assert not pywhy_graphs.has_adc(out_mag)
    out_edges = out_mag.edges()
    dir_edges = list(out_edges["directed"])
    assert (
        ("G", "S") in dir_edges
        and ("G", "L") in dir_edges
        and ("S", "L") in dir_edges
        and ("I", "S") in dir_edges
        and ("P", "S") in dir_edges
        and len(dir_edges) == 5
    )


def test_is_maximal():
    # X <- Y <-> Z <-> H; Z -> X
    admg = ADMG()
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("Z", "X", admg.directed_edge_name)
    admg.add_edge("Z", "Y", admg.bidirected_edge_name)
    admg.add_edge("Z", "H", admg.bidirected_edge_name)

    S = {}
    L = {"Y"}
    assert not pywhy_graphs.is_maximal(admg, L, S)


def test_all_vstructures():
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (3, 2), (4, 2)])

    # Generate the v-structures
    v_structs_edges = all_vstructures(G, as_edges=True)
    v_structs_tuples = all_vstructures(G, as_edges=False)

    # Assert that the returned values are as expected
    assert len(v_structs_edges) == 3
    assert len(v_structs_tuples) == 3
    assert (1, 2) in v_structs_edges or (2, 1) in v_structs_edges
    assert (3, 2) in v_structs_edges or (2, 3) in v_structs_edges
    assert (1, 2, 3) in v_structs_tuples or (3, 2, 1) in v_structs_tuples
    assert (4, 2, 3) in v_structs_tuples or (3, 2, 4) in v_structs_tuples

    G.remove_node(2)
    # Generate the v-structures
    v_structs_edges = all_vstructures(G, as_edges=True)
    v_structs_tuples = all_vstructures(G, as_edges=False)

    # Assert that the returned values are as expected
    assert len(v_structs_edges) == 0
    assert len(v_structs_tuples) == 0


def test_proper_possibly_directed():
    # X <- Y <-> Z <-> H; Z -> X

    admg = ADMG()
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "H", admg.directed_edge_name)

    Y = {"H"}
    X = {"Y"}

    correct = {("Y", "X", "Z", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "X", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "H", admg.directed_edge_name)

    Y = {"H"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "H"), ("A", "X", "Z", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("X", "A", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "H", admg.directed_edge_name)

    Y = {"H"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("X", "A", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "H", admg.directed_edge_name)
    admg.add_edge("K", "Z", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "X", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "H", admg.directed_edge_name)
    admg.add_edge("Z", "K", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {
        ("Y", "X", "Z", "K"),
        ("A", "X", "Z", "K"),
        ("Y", "X", "Z", "H"),
        ("A", "X", "Z", "H"),
    }
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "G", admg.directed_edge_name)
    admg.add_edge("G", "C", admg.directed_edge_name)
    admg.add_edge("C", "H", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "K", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "K"), ("A", "G", "C", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "G", admg.directed_edge_name)
    admg.add_edge("G", "C", admg.directed_edge_name)
    admg.add_edge("C", "H", admg.directed_edge_name)
    admg.add_edge("Z", "C", admg.directed_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "K", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "K"), ("Y", "X", "Z", "C", "H"), ("A", "G", "C", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "G", admg.directed_edge_name)
    admg.add_edge("A", "H", admg.directed_edge_name)
    admg.add_edge("K", "G", admg.directed_edge_name)
    admg.add_edge("K", "H", admg.directed_edge_name)

    Y = {"G", "H"}
    X = {"A", "K"}

    correct = {("K", "H"), ("K", "G"), ("A", "G"), ("A", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "G", admg.directed_edge_name)
    admg.add_edge("G", "C", admg.directed_edge_name)
    admg.add_edge("C", "H", admg.directed_edge_name)
    admg.add_edge("Z", "C", admg.bidirected_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "K", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {
        ("A", "G", "C", "H"),
        ("Y", "X", "Z", "K"),
    }
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out

    admg = ADMG()
    admg.add_edge("A", "G", admg.directed_edge_name)
    admg.add_edge("G", "C", admg.directed_edge_name)
    admg.add_edge("C", "H", admg.directed_edge_name)
    admg.add_edge("Z", "C", admg.bidirected_edge_name)
    admg.add_edge("Y", "X", admg.directed_edge_name)
    admg.add_edge("X", "Z", admg.directed_edge_name)
    admg.add_edge("Z", "K", admg.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {("Y", "X", "Z", "K"), ("A", "G", "C", "H")}
    out = pywhy_graphs.proper_possibly_directed_path(admg, X, Y)
    assert correct == out


def test_ppdp_PAG():

    pag = PAG()
    pag.add_edge("A", "G", pag.directed_edge_name)
    pag.add_edge("G", "C", pag.directed_edge_name)
    pag.add_edge("C", "H", pag.directed_edge_name)
    pag.add_edge("Z", "C", pag.circle_edge_name)
    pag.add_edge("C", "Z", pag.circle_edge_name)
    pag.add_edge("Y", "X", pag.directed_edge_name)
    pag.add_edge("X", "Z", pag.directed_edge_name)
    pag.add_edge("Z", "K", pag.directed_edge_name)

    Y = {"H", "K"}
    X = {"Y", "A"}

    correct = {
        ("Y", "X", "Z", "K"),
        ("Y", "X", "Z", "C", "H"),
        ("A", "G", "C", "H"),
        ("A", "G", "C", "Z", "K"),
    }
    out = pywhy_graphs.proper_possibly_directed_path(pag, X, Y)
    assert correct == out
