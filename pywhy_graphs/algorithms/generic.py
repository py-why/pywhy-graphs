from typing import List, Union

import networkx as nx

from pywhy_graphs import ADMG, CPDAG, PAG, StationaryTimeSeriesCPDAG, StationaryTimeSeriesPAG

from ..config import EdgeType
from ..typing import Node

__all__ = [
    "single_source_shortest_mixed_path",
    "is_node_common_cause",
    "set_nodes_as_latent_confounders",
    "is_valid_mec_graph",
    "inducing_path",
]


def is_node_common_cause(G: nx.DiGraph, node: Node, exclude_nodes: List[Node] = None) -> bool:
    """Check if a node is a common cause within the graph.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.
    node : node
        A node in the graph.
    exclude_nodes : list, optional
        Set of nodes to exclude from consideration, by default None.

    Returns
    -------
    is_common_cause : bool
        Whether or not the node is a common cause or not.
    """
    if exclude_nodes is None:
        exclude_nodes = []

    successors = G.successors(node)
    count = 0
    for succ in successors:
        if succ not in exclude_nodes:
            count += 1
        if count >= 2:
            return True
    return False


def set_nodes_as_latent_confounders(G: Union[nx.DiGraph, ADMG], nodes: List[Node]) -> ADMG:
    """Set nodes as latent unobserved confounders.

    Note that this only works if the original node is a common cause
    of some variables in the graph.

    Parameters
    ----------
    G : DiGraph
        A networkx DiGraph.
    nodes : list
        A list of nodes to set. They must all be common causes of
        variables within the graph.

    Returns
    -------
    graph : ADMG
        The mixed-edge causal graph that results.
    """
    bidirected_edges = []
    new_parent_ch_edges = []

    for node in nodes:
        # check if the node is a common cause
        if not is_node_common_cause(G, node, exclude_nodes=nodes):
            raise RuntimeError(
                f"{node} is not a common cause within the graph "
                f"given excluding variables. This function will only convert common "
                f"causes to latent confounders."
            )

        # keep track of which nodes to form c-components over
        successor_nodes = G.successors(node)
        for idx, succ in enumerate(successor_nodes):
            # TODO: do we want this?; add parent -> successor edges
            # if there are parents to this node, they must now point to all the successors
            for parent in G.predecessors(node):
                new_parent_ch_edges.append((parent, succ))

            # form a c-component among the successors
            if idx == 0:
                prev_succ = succ
                continue
            bidirected_edges.append((prev_succ, succ))
            prev_succ = succ

    # create the graph with nodes excluding those that are converted to latent confounders
    if isinstance(G, ADMG):
        new_graph = G.copy()
    elif isinstance(G, nx.DiGraph):
        new_graph = ADMG(G.copy())
    new_graph.remove_nodes_from(nodes)

    # create the c-component structures
    new_graph.add_edges_from(bidirected_edges, new_graph.bidirected_edge_name)

    # add additional edges that need to be accounted for
    new_graph.add_edges_from(new_parent_ch_edges, new_graph.directed_edge_name)
    return new_graph


def is_valid_mec_graph(G, on_error: str = "raise") -> bool:
    """Check G is a valid PAG.

    A valid CPDAG/PAG is one where each pair of nodes have
    at most one edge between them and the internal graph of directed edges
    do not form cycles.

    Parameters
    ----------
    G : PAG | CPDAG
        The PAG or CPDAG.
    on_error : str
        Whether to raise an error if the graph is non-compliant. Default is 'raise'.
        Other options are 'ignore'.

    Returns
    -------
    bool
        Whether G is a valid PAG or CPDAG.

    Notes
    -----
    This function does not check whether or not the PAG, or CPDAG are valid in the sense
    that the construction of the PAG/CPDAG was constrained to not contain any
    directed edge cycles.
    """
    if isinstance(G, CPDAG) or isinstance(G, StationaryTimeSeriesCPDAG):
        check_func = _check_adding_cpdag_edge
    elif isinstance(G, PAG) or isinstance(G, StationaryTimeSeriesPAG):
        check_func = _check_adding_pag_edge

    for edge_type, edgeview in G.edges().items():
        for u, v in edgeview:
            check_func(G, u, v, edge_type)
    return True


def _check_adding_cpdag_edge(graph: CPDAG, u_of_edge: Node, v_of_edge: Node, edge_type: EdgeType):
    """Check compatibility among internal graphs when adding an edge of a certain type.

    Parameters
    ----------
    graph : CPDAG
        The CPDAG we are adding edges to.
    u_of_edge : node
        The start node.
    v_of_edge : node
        The end node.
    edge_type : EdgeType
        The edge type that is being added.
    """
    raise_error = False
    if edge_type == EdgeType.DIRECTED.value:
        # there should not be a undirected edge, or a bidirected edge
        if graph.has_edge(u_of_edge, v_of_edge, graph.undirected_edge_name):
            raise_error = True
        if graph.has_edge(v_of_edge, u_of_edge, graph.directed_edge_name):
            raise RuntimeError(
                f"There is an existing {v_of_edge} -> {u_of_edge}. You are "
                f"trying to add a directed edge from {u_of_edge} -> {v_of_edge}. "
            )
    elif edge_type == EdgeType.UNDIRECTED.value:
        # there should not be any type of edge between the two
        if graph.has_edge(u_of_edge, v_of_edge, graph.directed_edge_name) or graph.has_edge(
            v_of_edge, u_of_edge, graph.directed_edge_name
        ):
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
        # there should not be a circle edge in the same direction, or a bidirected edge
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


def _directed_sub_graph_parents(G, node):
    """Finds the parents of a node in the directed subgraph.

    Parameters
    ----------
    G : Graph
        The graph.
    node : node label
        The node for which we have to find the parents.

    Returns
    -------
    out : set
        The parents of the provided node.
    """
    directed_parents = set(G.sub_directed_graph().predecessors(node))

    return directed_parents


def _bidirected_sub_graph_parents(G, node):
    """Finds the parents of a node in the bidirected subgraph.

    Parameters
    ----------
    G : Graph
        The graph.
    node : node label
        The node for which we have to find the parents.

    Returns
    -------
    out : set
        The parents of the provided node.
    """
    bidirected_parents = set()

    if not isinstance(G, CPDAG):
        bidirected_parents = set(G.sub_bidirected_graph().neighbors(node))

    return bidirected_parents


def _is_collider(G, prev_node, cur_node, node):
    """Checks if the given node is a collider or not.

    Parameters
    ----------
    G : graph
        The graph.
    node : node
        The node to be checked.

    Returns
    -------
    iscollider : bool
        Bool is set true if the node is a collider, false otherwise.
    """
    parents = _directed_sub_graph_parents(G, cur_node)
    parents = parents.union(_bidirected_sub_graph_parents(G, cur_node))

    if len(parents) == 2 and parents == {prev_node, node}:
        return True

    return False


def _recursive_path(G, node_x, node_y, L, S, visited, xyancestors, cur_node, prev_node):
    """Recursively explores a graph to find a path.

       Finds path that are compliant with the inducing path requirements.

    Parameters
    ----------
    G : graph
        The graph.
    node_x : node
        The source node.
    node_y : node
        The destination node
    L : set
        Set containing all the non-colliders.
    S : set
        Set containing all the colliders.
    visited : set
        Set containing all the nodes already visited.
    xyancestors : set
        Set containing the ancestors of X and Y.
    cur_node : node
        The current node.

    Returns
    -------
    path : Tuple[bool, path]
        A tuple containing a bool and a path which is empty if the bool is false.
    """
    path_exists = False
    path = []
    visited.add(cur_node)
    children = G.neighbors(cur_node)
    if cur_node is node_y:
        return (True, [node_y])
    for elem in children:
        if elem in visited:
            continue
        else:
            if (
                _is_collider(G, prev_node, cur_node, elem)
                and (cur_node not in S)
                and (cur_node not in xyancestors)
                and (cur_node is not node_y)
            ):
                continue
            elif (
                not _is_collider(G, prev_node, cur_node, elem)
                and (cur_node not in L)
                and (cur_node is not node_y)
            ):
                continue

            path_exists, temp_path = _recursive_path(
                G, node_x, node_y, L, S, visited, xyancestors, elem, cur_node
            )
            if path_exists:
                path.append(cur_node)
                path.extend(temp_path)
                break
    return (path_exists, path)


def inducing_path(G, node_x, node_y, L=None, S=None):
    """Checks if an inducing path exists between two nodes as defined in :footcite:`Zhang2008`.

    Parameters
    ----------
    G : Graph
        The graph.
    node_x : node
        The source node.
    node_y : node
        The destination node.
    L : set
        Nodes that are ignored on the path. Defaults to an empty set. See Notes for details.
    S:  set
        Nodes that are always conditioned on. Defaults to an empty set. See Notes for details.

    Returns
    -------
    path : Tuple[bool, path]
        A tuple containing a bool and a path if the bool is true, an empty list otherwise.

    Notes
    -----
    <TBD>
    """
    if L is None:
        L = set()

    if S is None:
        S = set()

    nodes = set(G.nodes)

    if node_x not in nodes or node_y not in nodes:
        raise ValueError("The provided nodes are not in the graph.")

    if node_x == node_y:
        raise ValueError("The start and destination nodes are the same.")

    path = []  # this will contain the path.

    xanc = _directed_sub_graph_parents(G, node_x)
    yanc = _directed_sub_graph_parents(G, node_y)

    xyancestors = xanc.union(yanc)
    children = G.neighbors(node_x)

    path_exists = False
    for elem in children:

        visited = {node_x}
        if elem not in visited:
            path_exists, temp_path = _recursive_path(
                G, node_x, node_y, L, S, visited, xyancestors, elem, node_x
            )
            if path_exists:
                path.append(node_x)
                path.extend(temp_path)
                break

    return (path_exists, path)
