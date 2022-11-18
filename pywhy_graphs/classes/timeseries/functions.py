import networkx as nx

from pywhy_graphs.classes.timeseries.base import BaseTimeSeriesDiGraph


def get_summary_graph(G: BaseTimeSeriesDiGraph) -> nx.DiGraph:
    # compute the summary graph
    summary_G = nx.DiGraph()

    # add all variables in ts-graph as nodes
    summary_G.add_nodes_from(G.variables)
