from typing import List, Optional, Tuple


from pywhy_graphs.classes.timeseries import TimeSeriesDiGraph
from pywhy_graphs.typing import Node


def _draw_pag_edges(
    dot,
    directed_edges=None,
    circle_edges: List[Tuple] = None,
    undirected_edges=None,
    bidirected_edges=None,
    **attrs,
):
    # keep track of edges with circular edges between each other because we want to
    # draw edges correctly when there are circular edges
    found_circle_sibs = set()

    # draw all possible causal edges on a graph
    if circle_edges is not None:
        for sib1, sib2 in circle_edges:
            # memoize if we have seen the bidirected circular edge before
            if f"{sib1}-{sib2}" in found_circle_sibs or f"{sib2}-{sib1}" in found_circle_sibs:
                continue
            found_circle_sibs.add(f"{sib1}-{sib2}")

            # set directionality of the edges
            direction = "forward"

            # check if the circular edge is bidirectional
            if (sib2, sib1) in circle_edges:
                direction = "both"
                arrowtail = "odot"
            elif directed_edges is not None and (sib2, sib1) in directed_edges:
                direction = "both"
                arrowtail = "normal"
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(
                sib1,
                sib2,
                arrowhead="odot",
                arrowtail=arrowtail,
                dir=direction,
                color="green",
                **attrs,
            )

    if undirected_edges is not None:
        for neb1, neb2 in undirected_edges:
            neb1, neb2 = str(neb1), str(neb2)
            dot.edge(neb1, neb2, dir="none", color="brown", **attrs)

    if bidirected_edges is not None:
        for sib1, sib2 in bidirected_edges:
            sib1, sib2 = str(sib1), str(sib2)
            dot.edge(sib1, sib2, dir="both", color="red", **attrs)
    return dot, found_circle_sibs


def draw_tsgraph(
    G: TimeSeriesDiGraph,
    direction: Optional[str] = None,
    node_order: Optional[List[Node]] = None,
    **attrs,
):
    """Visualize stationary time-series graphs.

    Here, time-series graph is drawn from left to right, where the left-most side
    represents the maximum-lag nodes and edges, and then as we move towards the right,
    the lag decreases.

    Only the compact ts-DMG is visualized.

    Parameters
    ----------
    G : BaseTimeSeriesDiGraph
        _description_
    direction : Optional[str], optional
        _description_, by default None
    """
    from graphviz import Digraph

    dot = Digraph()

    max_lag = G.max_lag

    # set direction from left to right if that's preferred
    if direction == "LR":
        dot.graph_attr["rankdir"] = direction

    dot.graph_attr["ordering"] = "out"

    # first define the node order if None is given
    if node_order is None:
        node_order = list(G.variables)

    shape = "square"  # 'plaintext'

    for variable in node_order:
        # start at max-lag
        for lag in range(max_lag, -1, -1):
            # first draw node
            node = f"({variable}, {-lag})"
            dot.node(node, shape=shape, height=".5", width=".5")

            # draw all lagged edges wrt this node
            lagged_nbrs = G.lagged_neighbors((variable, -lag))
            for nbr in lagged_nbrs:
                parent = f"{nbr}"
                dot.edge(parent, node, color="blue", **attrs)
    return dot


def draw(G, direction: Optional[str] = None, pos: Optional[dict] = None, **attrs):
    """Visualize the graph.

    Parameters
    ----------
    G : pywhy_nx.MixedEdgeGraph
        The mixed edge graph.
    direction : str, optional
        The direction, by default None.
    pos : dict, optional
        The positions of the nodes keyed by node with (x, y) coordinates as values.
        By default None, which will
        use the default layout from graphviz.
    attrs : dict
        Any additional edge attributes (must be strings). For more
        information, see documentation for GraphViz.

    Returns
    -------
    dot : graphviz Digraph
        DOT language representation of the graph.
    """
    from graphviz import Digraph

    dot = Digraph()

    # set direction from left to right if that's preferred
    if direction == "LR":
        dot.graph_attr["rankdir"] = direction

    shape = "square"  # 'plaintext'

    circle_edges = None
    directed_edges = None
    undirected_edges = None
    bidirected_edges = None
    if hasattr(G, "circle_edges"):
        circle_edges = G.circle_edges
    if hasattr(G, "directed_edges"):
        directed_edges = G.directed_edges
    if hasattr(G, "undirected_edges"):
        undirected_edges = G.undirected_edges
    if hasattr(G, "bidirected_edges"):
        bidirected_edges = G.bidirected_edges

    # draw PAG edges and keep track of the circular endpoints found
    dot, found_circle_sibs = _draw_pag_edges(
        dot,
        directed_edges,
        circle_edges=circle_edges,
        undirected_edges=undirected_edges,
        bidirected_edges=bidirected_edges,
    )

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
                dot.edge(parent, child, style="invis", **attrs)
            else:
                dot.edge(parent, child, color="blue", **attrs)

    return dot
