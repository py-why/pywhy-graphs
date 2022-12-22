import networkx as nx
import numpy as np
import pytest
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from numpy.testing import assert_array_equal

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.array.export import clearn_arr_to_graph, graph_to_arr


def create_clearn_nodes(n_nodes):
    n_nodes = 5
    nodes = []
    node_names = np.arange(n_nodes)
    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)
    return nodes


def dag():
    n_nodes = 5
    nodes = create_clearn_nodes(n_nodes)

    # 0 -> 1; 3 -> 4 <- 2
    # create the causal-learn graph and the
    # expected networkx-based graph
    G = GeneralGraph(nodes)
    G.add_directed_edge(G.nodes[0], G.nodes[1])
    G.add_directed_edge(G.nodes[3], G.nodes[4])
    G.add_directed_edge(G.nodes[2], G.nodes[4])
    edges = [
        (0, 1),
        (3, 4),
        (2, 4),
    ]
    expected_graph = nx.DiGraph(edges)
    return G, expected_graph


def admg():
    n_nodes = 5
    nodes = create_clearn_nodes(n_nodes)

    # 0 -> 1; 3 -> 4 <- 2
    # 1 <--> 0
    # create the causal-learn graph and the
    # expected networkx-based graph
    G = GeneralGraph(nodes)
    G.add_directed_edge(G.nodes[0], G.nodes[1])
    G.add_directed_edge(G.nodes[3], G.nodes[4])
    G.add_directed_edge(G.nodes[2], G.nodes[4])

    # add bidirected edges
    edge = Edge(G.nodes[0], G.nodes[1], Endpoint.ARROW, Endpoint.ARROW)
    G.add_edge(edge)

    directed_edges = nx.DiGraph(
        [
            (0, 1),
            (3, 4),
            (2, 4),
        ]
    )
    bidirected_edges = nx.Graph([(0, 1)])
    expected_graph = pywhy_graphs.ADMG(directed_edges, bidirected_edges)
    return G, expected_graph


def cpdag():
    n_nodes = 5
    nodes = create_clearn_nodes(n_nodes)

    # 0 -> 1; 3 -> 4 <- 2
    # 1 -- 3
    # create the causal-learn graph and the
    # expected networkx-based graph
    G = GeneralGraph(nodes)
    G.add_directed_edge(G.nodes[0], G.nodes[1])
    G.add_directed_edge(G.nodes[3], G.nodes[4])
    G.add_directed_edge(G.nodes[2], G.nodes[4])

    # add undirected edges
    edge = Edge(G.nodes[1], G.nodes[3], Endpoint.TAIL, Endpoint.TAIL)
    G.add_edge(edge)

    directed_edges = nx.DiGraph(
        [
            (0, 1),
            (3, 4),
            (2, 4),
        ]
    )
    undirected_edges = nx.Graph([(1, 3)])
    expected_graph = pywhy_graphs.CPDAG(directed_edges, undirected_edges)
    return G, expected_graph


def pag():
    n_nodes = 5
    nodes = create_clearn_nodes(n_nodes)

    G = GeneralGraph(nodes)
    # 0 -> 1; 3 -> 4 <- 2
    # 1 <--> 3
    # 0 o-o 3
    G.add_directed_edge(G.nodes[0], G.nodes[1])
    G.add_directed_edge(G.nodes[3], G.nodes[4])
    G.add_directed_edge(G.nodes[2], G.nodes[4])

    # add bidirected edges
    edge = Edge(G.nodes[1], G.nodes[3], Endpoint.ARROW, Endpoint.ARROW)
    G.add_edge(edge)

    # add circle edges
    edge = Edge(G.nodes[0], G.nodes[3], Endpoint.CIRCLE, Endpoint.CIRCLE)
    G.add_edge(edge)

    directed_edges = nx.DiGraph(
        [
            (0, 1),
            (3, 4),
            (2, 4),
        ]
    )
    bidirected_edges = nx.Graph([(1, 3)])
    circle_edges = nx.DiGraph(
        [
            (0, 3),
            (3, 0),
        ]
    )
    expected_graph = pywhy_graphs.PAG(
        directed_edges,
        incoming_bidirected_edges=bidirected_edges,
        incoming_circle_edges=circle_edges,
    )
    return G, expected_graph


def test_graph_to_arr_roundtrip_dag():
    clearn_G, expected_G = dag()
    graph_type = "dag"

    # get causal-learn graph as array
    arr = clearn_G.graph

    # get the node names in order of the array
    nodes = [node.get_name() for node in clearn_G.nodes]

    # convert array to networkx/pywhy-graphs graph
    graph = clearn_arr_to_graph(arr, arr_idx=nodes, graph_type=graph_type)
    assert nx.is_isomorphic(graph, expected_G)


@pytest.mark.parametrize(
    "clearn_G, expected_G, graph_type",
    [
        [*admg(), "admg"],
        [*cpdag(), "cpdag"],
        [*pag(), "pag"],
    ],
)
def test_graph_to_arr_roundtrip(
    clearn_G: GeneralGraph, expected_G: pywhy_nx.MixedEdgeGraph, graph_type
):
    # get causal-learn graph as array
    arr = clearn_G.graph

    # get the node names in order of the array
    nodes = [node.get_name() for node in clearn_G.nodes]

    # convert array to networkx/pywhy-graphs graph
    graph = clearn_arr_to_graph(arr, arr_idx=nodes, graph_type=graph_type)
    for edge_type, subG in graph.get_graphs().items():
        assert edge_type in expected_G.edge_types
        assert nx.is_isomorphic(subG, expected_G.get_graphs(edge_type))

    # convert graph back to array and it should match up to an order
    test_arr, arr_idx = graph_to_arr(graph, format="causal-learn")
    new_order = np.searchsorted(nodes, arr_idx)
    arr_idx = [arr_idx[idx] for idx in new_order]
    test_arr = test_arr[np.ix_(new_order, new_order)]
    assert arr_idx == nodes
    assert_array_equal(arr, test_arr)

    # explicitly passing in node order should make this work too
    test_arr, arr_idx = graph_to_arr(graph, format="causal-learn", node_order=nodes)
    assert_array_equal(arr, test_arr)
    assert set(arr_idx) == set(nodes)


def test_convert_clearn_errors():
    clearn_G, _ = dag()

    # get the node names in order of the array
    nodes = [node.get_name() for node in clearn_G.nodes]

    # only square arrays are acceptable
    arr = np.zeros((5, 3))
    with pytest.raises(RuntimeError, match="Only square arrays"):
        clearn_arr_to_graph(arr, arr_idx=nodes, graph_type="dag")

    # array idx and array should have same length
    arr = clearn_G.graph
    with pytest.raises(RuntimeError, match="The number of node names"):
        clearn_arr_to_graph(arr, arr_idx=nodes + ["test"], graph_type="dag")

    arr[0, 1] = 52
    with pytest.raises(RuntimeError, match="Some entries of array"):
        clearn_arr_to_graph(arr, arr_idx=nodes, graph_type="dag")
