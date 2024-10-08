from itertools import combinations
from typing import List, Optional, Set, Union

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
    "has_adc",
    "valid_mag",
    "dag_to_mag",
    "is_maximal",
    "all_vstructures",
    "proper_possibly_directed_path",
]


def is_node_common_cause(
    G: nx.DiGraph, node: Node, exclude_nodes: Optional[List[Node]] = None
) -> bool:
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


def _directed_sub_graph_ancestors(G, node: Node):
    """Finds the ancestors of a node in the directed subgraph.

    Parameters
    ----------
    G : Graph
        The graph.
    node : Node
        The node for which we have to find the ancestors.

    Returns
    -------
    out : set
        The parents of the provided node.
    """

    return nx.ancestors(G.sub_directed_graph(), node)


def _directed_sub_graph_parents(G, node: Node):
    """Finds the parents of a node in the directed subgraph.

    Parameters
    ----------
    G : Graph
        The graph.
    node : Node
        The node for which we have to find the parents.

    Returns
    -------
    out : set
        The parents of the provided node.
    """

    return set(G.sub_directed_graph().predecessors(node))


def _bidirected_sub_graph_neighbors(G, node: Node):
    """Finds the neighbors of a node in the bidirected subgraph.

    Parameters
    ----------
    G : Graph
        The graph.
    node : Node
        The node for which we have to find the neighbors.

    Returns
    -------
    out : set
        The parents of the provided node.
    """
    bidirected_parents = set()

    if not isinstance(G, CPDAG):
        bidirected_parents = set(G.sub_bidirected_graph().neighbors(node))

    return bidirected_parents


def _is_collider(G, prev_node: Node, cur_node: Node, next_node: Node):
    """Checks if the given node is a collider or not.

    Parameters
    ----------
    G : graph
        The graph.
    prev_node : node
        The previous node in the path.
    cur_node : node
        The node to be checked.
    next_node: Node
        The next node in the path.

    Returns
    -------
    iscollider : bool
        Bool is set true if the node is a collider, false otherwise.
    """
    parents = _directed_sub_graph_parents(G, cur_node)
    parents = parents.union(_bidirected_sub_graph_neighbors(G, cur_node))

    if prev_node in parents and next_node in parents:
        return True

    return False


def _shortest_valid_path(
    G,
    node_x: Node,
    node_y: Node,
    L: Set,
    S: Set,
    visited: Set,
    all_ancestors: Set,
    cur_node: Node,
    prev_node: Node,
):
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
    L : Set
        Set containing all the non-colliders.
    S : Set
        Set containing all the colliders.
    visited : Set
        Set containing all the nodes already visited.
    all_ancestors : Set
        Set containing all the ancestors a collider needs to be checked against.
    cur_node : node
        The current node.
    prev_node : node
        The previous node in the path.

    Returns
    -------
    path : Tuple[bool, path]
        A tuple containing a bool and a path which is empty if the bool is false.
    """
    path_exists = False
    path = []
    visited.add(cur_node)
    neighbors = G.neighbors(cur_node)

    if cur_node is node_y:
        return (True, [node_y])

    for elem in neighbors:
        if elem in visited:
            continue

        else:
            # If the current node is a collider, check that it is either an
            # ancestor of X, Y or any element of S or that it is
            # the destination node itself.
            if (
                _is_collider(G, prev_node, cur_node, elem)
                and (cur_node not in all_ancestors)
                and (cur_node not in S)
                and (cur_node is not node_y)
            ):
                continue

            # If the current node is not a collider, check that it is
            # either in L or the destination node itself.

            elif (
                not _is_collider(G, prev_node, cur_node, elem)
                and (cur_node not in L)
                and (cur_node is not node_y)
            ):
                continue

            # if it is a valid node and not the destination node,
            # check if it has a path to the destination node

            path_exists, temp_path = _shortest_valid_path(
                G, node_x, node_y, L, S, visited, all_ancestors, elem, cur_node
            )

            if path_exists:
                path.append(cur_node)
                path.extend(temp_path)
                break

    return (path_exists, path)


def inducing_path(G, node_x: Node, node_y: Node, L: Optional[Set] = None, S: Optional[Set] = None):
    """Checks if an inducing path exists between two nodes.

    An inducing path is defined in :footcite:`Zhang2008`.

    Parameters
    ----------
    G : Graph
        The graph.
    node_x : node
        The source node.
    node_y : node
        The destination node.
    L : Set
        Nodes that are ignored on the path. Defaults to an empty set. See Notes for details.
    S:  Set
        Nodes that are always conditioned on. Defaults to an empty set. See Notes for details.

    Returns
    -------
    path : Tuple[bool, path]
        A tuple containing a bool and a path if the bool is true, an empty list otherwise.

    Notes
    -----
    An inducing path intuitively is a path between two non-adjacent nodes that
    cannot be d-separated. Therefore, the path is always "active" regardless of
    what variables we condition on. L contains all the non-colliders, these nodes
    are ignored along the path. S contains nodes that are always conditioned on
    (hence if the ancestors of colliders are in S, then those collider
    paths are always "active").

    References
    ----------
    .. footbibliography::
    """
    if L is None:
        L = set()

    if S is None:
        S = set()

    nodes = set(G.nodes)

    if node_x not in nodes or node_y not in nodes:
        raise ValueError("The provided nodes are not in the graph.")

    if node_x == node_y:
        raise ValueError("The source and destination nodes are the same.")

    if (node_x in L) or (node_y in L) or (node_x in S) or (node_y in S):
        return (False, [])

    edges = G.edges()

    # XXX: fix this when graphs are refactored to only check for directed/bidirected edge types
    for elem in edges.keys():
        if elem not in {"directed", "bidirected"}:
            if len(edges[elem]) != 0:
                raise ValueError("Inducing Path is not defined for this graph.")

    path = []  # this will contain the path.

    x_ancestors = _directed_sub_graph_ancestors(G, node_x)
    y_ancestors = _directed_sub_graph_ancestors(G, node_y)

    xy_ancestors = x_ancestors.union(y_ancestors)

    s_ancestors: set[Node] = set()

    for elem in S:
        s_ancestors = s_ancestors.union(_directed_sub_graph_ancestors(G, elem))

    # ancestors of X, Y and all the elements of S

    all_ancestors = xy_ancestors.union(s_ancestors)
    x_neighbors = G.neighbors(node_x)

    path_exists = False
    for elem in x_neighbors:
        visited = {node_x}
        if elem not in visited:
            path_exists, temp_path = _shortest_valid_path(
                G, node_x, node_y, L, S, visited, all_ancestors, elem, node_x
            )
            if path_exists:
                path.append(node_x)
                path.extend(temp_path)
                break

    return (path_exists, path)


def has_adc(G):
    """Check if a graph has an almost directed cycle (adc).

    An almost directed cycle is a is a directed cycle containing
    one bidirected edge. For example, ``A -> B -> C <-> A`` is an adc.

    Parameters
    ----------
    G : Graph
        The graph.

    Returns
    -------
    adc_present : bool
        A boolean indicating whether an almost directed cycle is present or not.
    """

    adc_present = False

    biedges = G.bidirected_edges

    for elem in G.nodes:
        ancestors = nx.ancestors(G.sub_directed_graph(), elem)
        descendants = nx.descendants(G.sub_directed_graph(), elem)
        for elem in biedges:
            if (elem[0] in ancestors and elem[1] in descendants) or (
                elem[1] in ancestors and elem[0] in descendants
            ):  # there is a bidirected edge from one of the ancestors to a descendant
                return not adc_present

    return adc_present


def valid_mag(G: ADMG, L: Optional[set] = None, S: Optional[set] = None):
    """Checks if the provided graph is a valid maximal ancestral graph (MAG).

    A valid MAG as defined in :footcite:`Zhang2008` is a mixed edge graph that
    only has directed and bi-directed edges, no directed or almost directed
    cycles and no inducing paths between any two non-adjacent pair of nodes.

    Parameters
    ----------
    G : Graph
        The graph.

    Returns
    -------
    is_valid : bool
        A boolean indicating whether the provided graph is a valid MAG or not.

    """

    if L is None:
        L = set()

    if S is None:
        S = set()

    directed_sub_graph = G.sub_directed_graph()

    all_nodes = set(G.nodes)

    # check if there are any undirected edges or more than one edges b/w two nodes
    for node in all_nodes:
        nb = set(G.neighbors(node))
        for elem in nb:
            edge_data = G.get_edge_data(node, elem)
            if edge_data["undirected"] is not None:
                return False
            elif (edge_data["bidirected"] is not None) and (edge_data["directed"] is not None):
                return False

    # check if there are any directed cyclces
    try:
        nx.find_cycle(directed_sub_graph)  # raises a NetworkXNoCycle error
        return False
    except nx.NetworkXNoCycle:
        pass

    # check if there are any almost directed cycles
    if has_adc(G):  # if there is an ADC, it's not a valid MAG
        return False

    # check if there are any inducing paths between non-adjacent nodes

    for source in all_nodes:
        nb = set(G.neighbors(source))
        cur_set = all_nodes - nb
        cur_set.remove(source)
        for dest in cur_set:
            out = inducing_path(G, source, dest, L, S)
            if out[0] is True:
                return False

    return True


def dag_to_mag(G, L: Optional[Set] = None, S: Optional[Set] = None):
    """Converts a DAG to a valid MAG.

    The algorithm is defined in :footcite:`Zhang2008` on page 1877.

    Parameters:
    -----------
    G : Graph
        The graph.
    L : Set
        Nodes that are ignored on the path. Defaults to an empty set.
    S : Set
        Nodes that are always conditioned on. Defaults to an empty set.

    Returns
    -------
    mag : Graph
        The MAG.
    """

    if L is None:
        L = set()

    if S is None:
        S = set()

    # for each pair of nodes find if they have an inducing path between them.
    # only then will they be adjacent in the MAG.

    all_nodes = set(G.nodes)
    adj_nodes = []

    for source in all_nodes:
        copy_all = all_nodes.copy()
        copy_all.remove(source)
        for dest in copy_all:
            out = inducing_path(G, source, dest, L, S)
            if out[0] is True and {source, dest} not in adj_nodes:
                adj_nodes.append({source, dest})

    # find the ancestors of B U S (ansB) and A U S (ansA) for each pair of adjacent nodes

    mag = ADMG()

    for A, B in adj_nodes:
        AuS = S.union(A)
        BuS = S.union(B)

        ansA: Set = set()
        ansB: Set = set()

        for node in AuS:
            ansA = ansA.union(_directed_sub_graph_ancestors(G, node))

        for node in BuS:
            ansB = ansB.union(_directed_sub_graph_ancestors(G, node))

        if A in ansB and B not in ansA:
            # if A is in ansB and B is not in ansA, A -> B
            mag.add_edge(A, B, mag.directed_edge_name)

        elif A not in ansB and B in ansA:
            # if B is in ansA and A is not in ansB, A <- B
            mag.add_edge(B, A, mag.directed_edge_name)

        elif A not in ansB and B not in ansA:
            # if A is not in ansB and B is not in ansA, A <-> B
            mag.add_edge(B, A, mag.bidirected_edge_name)

        elif A in ansB and B in ansA:
            # if A is in ansB and B is in ansA, A - B
            mag.add_edge(B, A, mag.undirected_edge_name)

    return mag


def is_maximal(G, L: Optional[Set] = None, S: Optional[Set] = None):
    """Checks to see if the graph is maximal.

    Parameters:
    -----------
    G : Graph
        The graph.

    Returns
    -------
    is_maximal : bool
        A boolean indicating whether the provided graph is maximal or not.
    """

    if L is None:
        L = set()

    if S is None:
        S = set()

    all_nodes = set(G.nodes)
    checked = set()
    for source in all_nodes:
        nb = set(G.neighbors(source))
        cur_set = all_nodes - nb
        cur_set.remove(source)
        for dest in cur_set:
            current_pair = frozenset({source, dest})
            if current_pair not in checked:
                checked.add(current_pair)
                out = inducing_path(G, source, dest, L, S)
                if out[0] is True:
                    return False
            else:
                continue
    return True


def all_vstructures(G: nx.DiGraph, as_edges: bool = False):
    """Generate all v-structures in the graph.

    Parameters
    ----------
    G : DiGraph
        A directed graph.
    as_edges : bool
        Whether to return the v-structures as edges or as a set of tuples.

    Returns
    -------
    vstructs : set
        If ``as_edges`` is True, a set of v-structures in the graph encoded as the
        (parent_1, child, parent_2) tuple with child being an unshielded collider.
        Otherwise, a set of tuples of the form (parent, child), which are part of
        v-structures in the graph.
    """
    vstructs = set()
    for node in G.nodes:
        for p1, p2 in combinations(G.predecessors(node), 2):
            if p1 not in G.predecessors(p2) and p2 not in G.predecessors(p1):
                if as_edges:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
                else:
                    vstructs.add((p1, node, p2))  # type: ignore
    return vstructs


def _check_back_arrow(G: ADMG, X, Y: set):
    """Retrieve all the neigbors of X that do not have
    an arrow pointing back to it.

    Parameters
    ----------
    G : DiGraph
        A directed graph.
    X : Node
    Y : Set
        A set of neigbors of X.

    Returns
    -------
    out : set
        A set of all the neighbors of X that do not have an arrow pointing
        back to it.
    """
    out = set()

    for elem in Y:
        if not (
            G.has_edge(X, elem, G.bidirected_edge_name) or G.has_edge(elem, X, G.directed_edge_name)
        ):
            out.update(elem)

    return out


def _get_neighbors_of_set(G, X: set):
    """Retrieve all the neigbors of X when X has more than one element.

    Note that if X is not a set, graph.neighbors(X) is sufficient.

    Parameters
    ----------
    G : DiGraph
        A directed graph.
    X : Set

    Returns
    -------
    out : set
        A set of all the neighbors of X.
    """

    out = set()

    for elem in X:
        elem_neighbors = set(G.neighbors(elem))
        elem_possible_neighbors = _check_back_arrow(G, elem, elem_neighbors)
        to_remove = X.intersection(elem_possible_neighbors)
        elem_neighbors = elem_possible_neighbors - to_remove

        if len(elem_neighbors) != 0:
            for nbh in elem_neighbors:
                temp = (elem,)
                temp = temp + (nbh,)
                out.add(temp)
    return out


def _recursively_find_pd_paths(G, X, paths, Y):
    """Recursively finds all the possibly directed paths for a given
    graph.

    Parameters
    ----------
    G : DiGraph
        A directed graph.
    X : Set
        Source.
    paths : Set
        Set of initial paths from X.
    Y : Set
        Destination

    Returns
    -------
    out : set
        A set of all the possibly directed paths.
    """

    counter = 0
    new_paths = set()

    for elem in paths:
        cur_elem = elem[-1]

        if cur_elem in Y:
            new_paths.add(elem)
            continue

        nbr_temp = G.neighbors(cur_elem)
        nbr_possible = _check_back_arrow(G, cur_elem, nbr_temp)

        if len(nbr_possible) == 0:
            new_paths = new_paths + (elem,)

        possible_end = nbr_possible.intersection(Y)

        if len(possible_end) != 0:
            for nbr in possible_end:
                temp_path = elem
                temp_path = temp_path + (nbr,)
                new_paths.add(temp_path)

        remaining_nodes = nbr_possible - possible_end
        remaining_nodes = (
            remaining_nodes
            - remaining_nodes.intersection(set(elem))
            - remaining_nodes.intersection(X)
        )

        temp_set = set()
        for nbr in remaining_nodes:
            temp_paths = elem
            temp_paths = temp_paths + (nbr,)
            temp_set.add(temp_paths)

        new_paths.update(_recursively_find_pd_paths(G, X, temp_set, Y))

    return new_paths


def proper_possibly_directed_path(G, X: Optional[Set], Y: Optional[Set]):
    """Find all the proper possibly directed paths in a graph. A proper possibly directed
    path from X to Y is a set of edges with just the first node in X and none of the edges
    with an arrow pointing back to X.

    Parameters
    ----------
    G : DiGraph
        A directed graph.
    X : Set
        Source.
    Y : Set
        Destination

    Returns
    -------
    out : set
        A set of all the proper possibly directed paths.

    Examples
    --------
    The function generates a set of tuples containing all the valid
    proper possibly directed paths from X to Y.

    >>> import pywhy_graphs
    >>> from pywhy_graphs import PAG
    >>> pag = PAG()
    >>> pag.add_edge("A", "G", pag.directed_edge_name)
    >>> pag.add_edge("G", "C", pag.directed_edge_name)
    >>> pag.add_edge("C", "H", pag.directed_edge_name)
    >>> pag.add_edge("Z", "C", pag.circle_edge_name)
    >>> pag.add_edge("C", "Z", pag.circle_edge_name)
    >>> pag.add_edge("Y", "X", pag.directed_edge_name)
    >>> pag.add_edge("X", "Z", pag.directed_edge_name)
    >>> pag.add_edge("Z", "K", pag.directed_edge_name)
    >>> Y = {"H", "K"}
    >>> X = {"Y", "A"}
    >>> pywhy_graphs.proper_possibly_directed_path(pag, X, Y)
    {('A', 'G', 'C', 'H'), ('Y', 'X', 'Z', 'C', 'H'), ('Y', 'X', 'Z', 'K'), ('A', 'G', 'C', 'Z', 'K')}

    """

    if isinstance(X, set):
        x_neighbors = _get_neighbors_of_set(G, X)
    else:
        nbr_temp = G.neighbors(X)
        nbr_possible = _check_back_arrow(nbr_temp)
        x_neighbors = []

        for elem in nbr_possible:
            temp = dict()
            temp[0] = X
            temp[1] = elem
            x_neighbors.append(temp)

    path_list = _recursively_find_pd_paths(G, X, x_neighbors, Y)

    return path_list
