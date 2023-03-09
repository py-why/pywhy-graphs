import networkx as nx
from ananke.graphs import ADMG, BG, CG, DAG, SG, UG, Graph

import pywhy_graphs
import pywhy_graphs.networkx as pywhy_nx


def graph_to_ananke(
    graph: pywhy_nx.MixedEdgeGraph,
    directed_edge_name="directed",
    bidirected_edge_name="bidirected",
    undirected_edge_name="undirected",
) -> Graph:
    """
    Convert causal graph to Ananke graph. Supports graphs with directed, undirected, and
    bidirected edges -- including DAGs, ADMGs, and CGs (chain graphs).

    Parameters
    ----------
    graph : pywhy_nx.MixedEdgeGraph
        The mixed edge causal graph
    directed_edge_name : str
        Name of the directed edge, default is directed.
    bidirected_edge_name : str
        Name of the bidirected edge, default is bidirected.
    undirected_edge_name : str
        Name of the undirected edge, default is undirected.


    Returns
    -------
    result : Graph
        The Ananke graph

    """
    vertices = graph.nodes
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
        result = CG(vertices, di_edges=di_edges, ud_edges=ud_edges)
    elif not has_directed and has_bidirected and not has_undirected:
        result = BG(vertices, bi_edges=bi_edges)
    elif not has_directed and not has_bidirected and has_undirected:
        result = UG(vertices, ud_edges=ud_edges)
    else:
        result = SG(vertices, di_edges=di_edges, bi_edges=bi_edges, ud_edges=ud_edges)

    return result


def ananke_to_graph(ananke_graph: Graph) -> pywhy_nx.MixedEdgeGraph:
    """
    Convert Ananke graph to causal graph.

    Parameters
    ----------
    ananke_graph : Graph
        The Ananke graph
    directed_edge_name : str
        Name of the directed edge, default is directed.
    bidirected_edge_name : str
        Name of the bidirected edge, default is bidirected.
    undirected_edge_name : str
        Name of the undirected edge, default is undirected.


    Returns
    -------
    result : pywhy_nx.MixedEdgeGraph
        The mixed edge graph.
    """
    bidirected_edge_name = "bidirected"
    directed_edge_name = "directed"
    undirected_edge_name = "undirected"
    if type(ananke_graph) == DAG:
        graph = pywhy_graphs.ADMG()
        graph.add_nodes_from(ananke_graph.vertices)
        graph.add_edges_from(ananke_graph.di_edges, edge_type=directed_edge_name)
    elif type(ananke_graph) == ADMG:
        graph = pywhy_graphs.ADMG()
        graph.add_nodes_from(ananke_graph.vertices)
        graph.add_edges_from(ananke_graph.di_edges, edge_type=directed_edge_name)
        graph.add_edges_from(ananke_graph.bi_edges, edge_type=bidirected_edge_name)
    else:
        graph = pywhy_nx.MixedEdgeGraph()
        graph.add_nodes_from(ananke_graph.vertices)
        if ananke_graph.di_edges:
            directed_edges = nx.DiGraph(ananke_graph.di_edges)
            graph.add_edge_type(directed_edges, directed_edge_name)
        if ananke_graph.bi_edges:
            bidirected_edges = nx.Graph(ananke_graph.bi_edges)
            graph.add_edge_type(bidirected_edges, bidirected_edge_name)
        if ananke_graph.ud_edges:
            undirected_edges = nx.Graph(ananke_graph.ud_edges)
            graph.add_edge_type(undirected_edges, undirected_edge_name)

    return graph
