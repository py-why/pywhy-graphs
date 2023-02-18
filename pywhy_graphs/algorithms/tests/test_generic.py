import networkx as nx

import pywhy_graphs


def test_convert_to_latent_confounder():
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = pywhy_graphs.ADMG(incoming_graph_data)

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
