import networkx as nx

from pywhy_graphs import CPDAG, PAG

from ..config import EdgeType
from ..typing import Node

__all__ = ["single_source_shortest_mixed_path"]


def _check_adding_cpdag_edge(graph: CPDAG, u_of_edge: Node, v_of_edge: Node, edge_type: EdgeType):
    """Check compatibility among internal graphs when adding an edge of a certain type.

    Parameters
    ----------
    u_of_edge : node
        The start node.
    v_of_edge : node
        The end node.
    edge_type : EdgeType
        The edge type that is being added.
    """
    raise_error = False
    if edge_type == EdgeType.DIRECTED:
        # there should not be a circle edge, or a bidirected edge
        if graph.has_edge(u_of_edge, v_of_edge, graph.undirected_edge_name):
            raise_error = True
        if graph.has_edge(v_of_edge, u_of_edge, graph.directed_edge_name):
            raise RuntimeError(
                f"There is an existing {v_of_edge} -> {u_of_edge}. You are "
                f"trying to add a directed edge from {u_of_edge} -> {v_of_edge}. "
                f"If your intention is to create a bidirected edge, first remove the "
                f"edge and then explicitly add the bidirected edge."
            )
    elif edge_type == EdgeType.UNDIRECTED:
        # there should not be any type of edge between the two
        if graph.has_edge(u_of_edge, v_of_edge):
            raise_error = True

    if raise_error:
        raise RuntimeError(
            f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
            f"Adding a {edge_type} edge is not possible. Please remove the existing "
            f"edge first."
        )


def _check_adding_pag_edge(graph: PAG, u_of_edge: Node, v_of_edge: Node, edge_type: EdgeType):
    """Check compatibility among internal graphs when adding an edge of a certain type.

    Parameters
    ----------
    u_of_edge : node
        The start node.
    v_of_edge : node
        The end node.
    edge_type : EdgeType
        The edge type that is being added.
    """
    raise_error = False
    if edge_type == EdgeType.ALL.value:
        if graph.has_edge(u_of_edge, v_of_edge):
            raise_error = True
    elif edge_type == EdgeType.CIRCLE.value:
        # there should not be an existing arrow
        # nor a bidirected arrow
        if graph.has_edge(u_of_edge, v_of_edge, graph.directed_edge_name) or graph.has_edge(
            u_of_edge, v_of_edge, graph.bidirected_edge_name
        ):
            raise_error = True
    elif edge_type == EdgeType.DIRECTED.value:
        # there should not be a circle edge, or a bidirected edge
        if graph.has_edge(u_of_edge, v_of_edge, graph.circle_edge_name) or graph.has_edge(
            u_of_edge, v_of_edge, graph.bidirected_edge_name
        ):
            raise_error = True
        if graph.has_edge(v_of_edge, u_of_edge, graph.directed_edge_name):
            raise RuntimeError(
                f"There is an existing {v_of_edge} -> {u_of_edge}. You are "
                f"trying to add a directed edge from {u_of_edge} -> {v_of_edge}. "
                f"If your intention is to create a bidirected edge, first remove the "
                f"edge and then explicitly add the bidirected edge."
            )
    elif edge_type == EdgeType.BIDIRECTED.value:
        # there should not be any type of edge between the two
        if (
            graph.has_edge(u_of_edge, v_of_edge, graph.directed_edge_name)
            or graph.has_edge(u_of_edge, v_of_edge, graph.circle_edge_name)
            or graph.has_edge(v_of_edge, u_of_edge, graph.directed_edge_name)
            or graph.has_edge(v_of_edge, u_of_edge, graph.circle_edge_name)
        ):
            raise_error = True

    if raise_error:
        raise RuntimeError(
            f"There is already an existing edge between {u_of_edge} and {v_of_edge}. "
            f"Adding a {edge_type} edge is not possible. Please remove the existing "
            f"edge first. {graph.edges()}"
        )


def single_source_shortest_mixed_path(G, source, cutoff=None, valid_path=None):
    """Compute shortest mixed-edge path between source and all other nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    valid_path : function
        Function to determine i

    Returns
    -------
    lengths : dictionary
        Dictionary, keyed by target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_shortest_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    The shortest path is not necessarily unique. So there can be multiple
    paths between the source and each target node, all of which have the
    same 'shortest' length. For each target node, this function returns
    only one of those paths.

    See Also
    --------
    shortest_path
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")

    def join(p1, p2):
        return p1 + p2

    if cutoff is None:
        cutoff = float("inf")
    if valid_path is None:
        valid_path = lambda *_: True

    nextlevel = {source: 1}  # list of nodes to check at next level
    paths = {source: [source]}  # paths dictionary  (paths to key from source)
    return dict(_single_shortest_path_early_stop(G, nextlevel, paths, cutoff, join, valid_path))


def _single_shortest_path_early_stop(G, firstlevel, paths, cutoff, join, valid_path):
    """Return shortest paths.

    Shortest Path helper function.

    Parameters
    ----------
    G : Graph
        Graph
    firstlevel : dict
        starting nodes, e.g. {source: 1} or {target: 1}
    paths : dict
        paths for starting nodes, e.g. {source: [source]}
    cutoff : int or float
        level at which we stop the process
    join : function
        function to construct a path from two partial paths. Requires two
        list inputs `p1` and `p2`, and returns a list. Usually returns
        `p1 + p2` (forward from source) or `p2 + p1` (backward from target)
    valid_path : function
        function to determine if the current path is a valid path.
        Input of graph, current node, and the next node. Returns true
        if continuing along the path of 'current node' *-* 'next node'
        is valid. Returns false otherwise and the path will be cut short.

    Returns
    -------
    paths : dict
        The updated paths for starting nodes.
    """
    level = 0  # the current level
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G.neighbors(v):
                if w not in paths and valid_path(G, v, w):
                    paths[w] = join(paths[v], [w])
                    nextlevel[w] = 1
        level += 1
    return paths
