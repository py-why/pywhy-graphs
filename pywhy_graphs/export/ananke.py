from typing import List, Optional, Tuple

import ananke
import networkx as nx
from ananke.graphs import DAG, Graph, ADMG, CG

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx


def graph_to_ananke(graph: pywhy_nx.MixedEdgeGraph) -> Graph:
    """
    Convert causal graph to Ananke graph. Supports DAGs, ADMGs


    """
    vertices = graph.nodes
    bidirected_edge_name = "bidirected"
    directed_edge_name = "directed"
    undirected_edge_name = "undirected"
    has_directed = False
    has_bidirected = False
    has_undirected = False
    for edge_type, sub_graph in graph.get_graphs().items():
        if sub_graph.edges:
            if edge_type == directed_edge_name:
                has_directed = True
                di_edges = [e for e in sub_graph.edges]
            elif edge_type == bidirected_edge_name:
                has_bidirected = True
                bi_edges = [e for e in sub_graph.edges]
            elif edge_type == undirected_edge_name:
                has_undirected = True
                ud_edges = [e for e in sub_graph.edges]
    if has_directed and not has_bidirected and not has_undirected:
        result = DAG(vertices, di_edges)
    elif has_directed and has_bidirected and not has_undirected:
        result = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
    elif has_directed and not has_bidirected and has_undirected:
        result = CG(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)
    else:
        raise ValueError(graph.get_graphs().items(), has_directed, has_bidirected, has_undirected)

    return result


def ananke_to_graph(ananke_graph: Graph) -> pywhy_nx.MixedEdgeGraph:
    """
    Convert Ananke graph to causal graph.

    """
    bidirected_edge_name = "bidirected"
    directed_edge_name = "directed"
    undirected_edge_name = "undirected"
    if type(ananke_graph) == DAG:
        graph = nx.DiGraph()
        graph.add_nodes_from(ananke_graph.vertices)
        graph.add_edges_from(ananke_graph.di_edges)
    elif type(ananke_graph) == ADMG:
        graph = pywhy_graphs.ADMG()
        graph.add_nodes_from(ananke_graph.vertices)
        graph.add_edges_from(ananke_graph.di_edges, edge_type=directed_edge_name)
        graph.add_edges_from(ananke_graph.bi_edges, edge_type=bidirected_edge_name)
    elif type(ananke_graph) == CG:
        graph = pywhy_nx.MixedEdgeGraph()
        graph.add_nodes_from(ananke_graph.vertices)
        graph.add_edges_from(ananke_graph.di_edges, edge_type=directed_edge_name)
        graph.add_edges_from(ananke_graph.ud_edges, edge_type=undirected_edge_name)
    else:
        raise ValueError("unsupported ananke graph")

    return graph
