from typing import Optional

import networkx as nx

import pywhy_graphs as pg


def draw(G: nx.MixedEdgeGraph, directed_graph_name: str, direction: Optional[str] = None, **attrs):
    """Visualize the graph.

    Parameters
    ----------
    G : nx.MixedEdgeGraph
        The mixed edge graph with a directed subgraph.
    directed_graph_name : str
        The name of the directed edge subgraph.
    direction : str, optional
        The direction, by default None.
    attrs : dict
        Any additional edge attributes (must be strings). For more
        information, see documentation for GraphViz.

    Returns
    -------
    dot : Digraph
        dot language representation of the graph.
    """
    from graphviz import Digraph

    dot = Digraph()

    # set direction from left to right if that's preferred
    if direction == "LR":
        dot.graph_attr["rankdir"] = direction

    # get directed subgraph
    directed_G = G.get_graphs(directed_graph_name)

    # compute which edges have circular endpoints, so we only draw edges once
    # between any two nodes
    circle_edges = set()
    if hasattr(G, "circle_edges"):
        for sib1, sib2 in G.circle_edges:
            sib1, sib2 = str(sib1), str(sib2)
            circle_edges.add(frozenset([sib1, sib2]))

    for parent, child in directed_G.edges:
        if (parent, child) in circle_edges:
            raise RuntimeError(
                f"There cannot be an arrowhead and a circle edge from {parent} to {child}."
            )

        # arrowhead and circle-endpoint: child <-o parent
        if (child, parent) in circle_edges:
            # if 'odot' in circle_edges[child][parent]:
            dot.edge(parent, child, color="blue", arrowhead="normal", arrowtail="odot", **attrs)
            circle_edges.remove((child, parent))
        else:
            dot.edge(parent, child, color="blue", arrowhead="normal", **attrs)

    # now for all rest of circular edges, add them in
    for u, v in circle_edges:
        # u o-o v
        if (v, u) in circle_edges:
            dot.edge(u, v, arrowhead="odot", arrowtail="odot", color="green", **attrs)
        # u -o v
        else:
            dot.edge(u, v, arrowhead="odot", color="green", **attrs)

    # draw undirected edges if they are present
    if hasattr(G, "undirected_edges"):
        undirected_edges = G.undirected_edges
    elif "undirected" in G.edge_types is not None:
        undirected_edges = G.get_graphs("undirected").edges
    else:
        undirected_edges = pg.CPDAG().undirected_edges
    for neb1, neb2 in undirected_edges:
        neb1, neb2 = str(neb1), str(neb2)
        dot.edge(neb1, neb2, dir="none", color="brown", **attrs)

    # draw bidirected edges if they are present
    if hasattr(G, "bidirected_edges"):
        bidirected_edges = G.bidirected_edges
    elif "bidirected" in G.edge_types is not None:
        bidirected_edges = G.get_graphs("bidirected").edges
    else:
        bidirected_edges = pg.CPDAG().undirected_edges
    for sib1, sib2 in bidirected_edges:
        sib1, sib2 = str(sib1), str(sib2)
        dot.edge(sib1, sib2, dir="both", color="red", **attrs)

    return dot
