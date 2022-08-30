import networkx as nx


def draw(G: nx.MixedEdgeGraph, direction=None):
    """
    Visualize the graph.

    :return : dot language representation of the graph.
    """
    from graphviz import Digraph

    dot = Digraph()

    # set direction from left to right if that's preferred
    if direction == "LR":
        dot.graph_attr["rankdir"] = direction

    shape = "square"  # 'plaintext'
    for v in G.nodes:
        child = str(v)

        dot.node(child, shape=shape, height=".5", width=".5")

        for parent in G.predecessors(v):
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

        if hasattr(G, "circle_edges"):
            for sib1, sib2 in G.circle_edges:
                sib1, sib2 = str(sib1), str(sib2)
                dot.edge(sib1, sib2, arrowhead="circle", color="green")

    return dot
