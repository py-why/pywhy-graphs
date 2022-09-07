from typing import List

import pywhy_graphs
from pywhy_graphs.typing import Node


def has_multiple_edges(G: pywhy_graphs.ADMG, u: Node, v: Node) -> bool:
    edge_count = 0
    for _, graph in G.get_graphs().items():
        if graph.has_edge(u, v):
            edge_count += 1

    return edge_count > 1


def edge_types(G: pywhy_graphs.ADMG, u: Node, v: Node) -> List[str]:
    edge_types = []
    for edge_type, graph in G.get_graphs().items():
        if graph.has_edge(u, v) or graph.has_edge(v, u):
            edge_types.append(edge_type)
    return edge_types
