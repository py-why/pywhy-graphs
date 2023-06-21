import networkx as nx
import pytest

import pywhy_graphs
from pywhy_graphs import ADMG


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
