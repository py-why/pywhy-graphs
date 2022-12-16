from typing import Optional

import networkx as nx


def draw(G: nx.MixedEdgeGraph, direction: Optional[str] = None, pos: Optional[dict] = None):
    """Visualize the graph.

    Parameters
    ----------
    G : nx.MixedEdgeGraph
        The mixed edge graph.
    direction : str, optional
        The direction, by default None.
    pos : dict, optional
        The positions of the nodes, by default None

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
    shape = "square"  # 'plaintext'

    found_circle_sibs = set()
    if hasattr(G, "circle_edges"):
        for sib1, sib2 in G.circle_edges:
            # memoize if we have seen the bidirected circular edge before
            if f"{sib1}-{sib2}" in found_circle_sibs or f"{sib2}-{sib1}" in found_circle_sibs:
                continue
            found_circle_sibs.add(f"{sib1}-{sib2}")

            # set directionality of the edges
            dir = "forward"
            if (sib2, sib1) in G.circle_edges:
                dir = "both"
                arrowtail = "odot"
            elif (sib2, sib1) in G.directed_edges:
                dir = "both"
                arrowtail = "normal"
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(sib1, sib2, arrowhead="odot", arrowtail=arrowtail, dir=dir, color="green")

    for v in G.nodes:
        child = str(v)
        if pos and pos.get(v) is not None:
            dot.node(child, shape=shape, height=".5", width=".5", pos=f"{pos[v][0]},{pos[v][1]}!")
        else:
            dot.node(child, shape=shape, height=".5", width=".5")

        for parent in G.predecessors(v):
            # memoize if we have seen the bidirected circular edge before
            if f"{child}-{parent}" in found_circle_sibs or f"{parent}-{child}" in found_circle_sibs:
                continue
            parent = str(parent)
            if parent == v:
                dot.edge(parent, child, style="invis")
            else:
                dot.edge(parent, child, color="blue")

    if hasattr(G, "undirected_edges"):
        for neb1, neb2 in G.undirected_edges:
            neb1, neb2 = str(neb1), str(neb2)
            dot.edge(neb1, neb2, dir="none", color="brown")

    if hasattr(G, "bidirected_edges"):
        for sib1, sib2 in G.bidirected_edges:
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(sib1, sib2, dir="both", color="red")

    return dot
