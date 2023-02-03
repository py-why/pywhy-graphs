import logging
from collections import deque
from itertools import chain
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from pywhy_graphs import PAG, StationaryTimeSeriesPAG
from pywhy_graphs.algorithms.generic import single_source_shortest_mixed_path
from pywhy_graphs.typing import Node, TsNode

logger = logging.getLogger()

__all__ = [
    "possible_ancestors",
    "possible_descendants",
    "discriminating_path",
    "pds",
    "pds_path",
    "uncovered_pd_path",
    "pds_t",
    "pds_t_path",
]


def _possibly_directed(G: PAG, i: Node, j: Node, reverse: bool = False):
    """Check that path is possibly directed.

    A possibly directed path is one of the form:
    - ``i -> j``
    - ``i o-> j``
    - ``i o-o j``

    Parameters
    ----------
    G : PAG
        The graph.
    i : Node
        The first node.
    j : Node
        The second node.
    reverse : bool
        Whether to check the reverse direction for valid path. If true,
        will check for ``i *-> j``. If false (default) will check for
        ``i <-* j``.

    Returns
    -------
    valid : bool
        Whether to path from ``... i *-* j ...`` is a valid path.
    """
    if i not in G.neighbors(j):
        return False

    if reverse:
        # i *-> j is invalid
        direct_check = G.has_edge(i, j, G.directed_edge_name)
    else:
        # i <-* j is invalid
        direct_check = G.has_edge(j, i, G.directed_edge_name)

    # the direct check checks for i *-> j or i <-* j
    # i <-> j is also checked
    # everything else is valid
    if direct_check or G.has_edge(i, j, G.bidirected_edge_name):
        return False
    return True


def possible_ancestors(G: PAG, source: Node) -> Set[Node]:
    """Possible ancestors of a source node.

    Parameters
    ----------
    G : PAG
        The graph.
    source : Node
        The source node to start at.

    Returns
    -------
    possible_ancestors : Set[Node]
        The set of nodes that are possible ancestors.
    """
    valid_path = lambda *args: _possibly_directed(*args, reverse=True)  # type: ignore
    # perform BFS starting at source using neighbors
    paths = single_source_shortest_mixed_path(G, source, valid_path=valid_path)
    return set(paths.keys())


def possible_descendants(G: PAG, source: Node) -> Set[Node]:
    """Possible descendants of a source node.

    Parameters
    ----------
    G : PAG
        The graph.
    source : Node
        The source node to start at.

    Returns
    -------
    possible_descendants : Set[Node]
        The set of nodes that are possible descendants.
    """
    valid_path = lambda *args: _possibly_directed(*args, reverse=False)  # type: ignore
    # perform BFS starting at source using neighbors
    paths = single_source_shortest_mixed_path(G, source, valid_path=valid_path)
    return set(paths.keys())


def is_definite_collider(G: PAG, node1: Node, node2: Node, node3: Node) -> bool:
    """Check if <node1, node2, node3> path forms a definite collider.

    I.e. node1 *-> node2 <-* node3.

    Parameters
    ----------
    node1 : node
        A node on the path to check.
    node2 : node
        A node on the path to check.
    node3 : node
        A node on the path to check.

    Returns
    -------
    is_collider : bool
        Whether or not the path is a definite collider.
    """
    # check arrow from node1 into node2
    condition_one = G.has_edge(node1, node2, G.directed_edge_name) or G.has_edge(
        node1, node2, G.bidirected_edge_name
    )

    # check arrow from node2 into node1
    condition_two = G.has_edge(node3, node2, G.directed_edge_name) or G.has_edge(
        node3, node2, G.bidirected_edge_name
    )
    return condition_one and condition_two


def is_definite_noncollider(G: PAG, node1: Node, node2: Node, node3: Node) -> bool:
    """Check if <node1, node2, node3> path forms a definite non-collider.

    I.e. node1 *-* node2 -> node3, or node1 <- node2 *-* node3

    Parameters
    ----------
    node1 : node
        A node on the path to check.
    node2 : node
        A node on the path to check.
    node3 : node
        A node on the path to check.

    Returns
    -------
    is_noncollider : bool
        Whether or not the path is a definite non-collider.
    """
    condition_one = G.has_edge(node2, node3, G.directed_edge_name) and not G.has_edge(
        node3, node2, G.directed_edge_name
    )
    condition_two = G.has_edge(node2, node1, G.directed_edge_name) and not G.has_edge(
        node1, node2, G.directed_edge_name
    )
    return condition_one or condition_two


def discriminating_path(
    graph: PAG, u: Node, a: Node, c: Node, max_path_length: Optional[int] = None
) -> Tuple[bool, List[Node], Set[Node]]:
    """Find the discriminating path for <..., a, u, c>.

    A discriminating path, p = <v, ..., a, u, c>, is one
    where:
    - p has at least 3 edges
    - u is non-endpoint and u is adjacent to c
    - v is not adjacent to c
    - every vertex between v and u is a collider on p and parent of c

    Parameters
    ----------
    graph : PAG
        PAG to orient.
    u : node
        A node in the graph.
    a : node
        A node in the graph.
    c : node
        A node in the graph.
    max_path_length : optional, int
        The maximum distance to check in the graph. By default None, which sets
        it to 1000.

    Returns
    -------
    explored_nodes : set
        A set of explored nodes.
    found_discriminating_path : bool
        Whether or not a discriminating path was found.
    disc_path : list
        The discriminating path starting from node c.
    """
    if max_path_length is None:
        max_path_length = 1000

    explored_nodes: Set[Node] = set()
    found_discriminating_path = False
    disc_path: List[Node] = []

    # parents of c form the discriminating path
    cparents = graph.parents(c)

    # keep track of the distance searched
    distance = 0

    # keep track of the previous nodes, i.e. to build a path
    # from node (key) to its child along the path (value)
    descendant_nodes = dict()
    descendant_nodes[u] = c
    descendant_nodes[a] = u

    # keep track of paths of certain nodes that were already explored
    # start off with the valid triple <a, u, c>
    # - u is adjacent to c
    # - u has an arrow pointing to a
    # - TBD a is a definite collider
    # - TBD endpoint is not adjacent to c
    explored_nodes.add(c)
    explored_nodes.add(u)
    explored_nodes.add(a)

    # a must be a parent of c
    if not graph.has_edge(a, c, graph.directed_edge_name):
        return found_discriminating_path, disc_path, explored_nodes

    # a and u must be connected by a bidirected edge, or with an edge towards a
    # for a to be a definite collider
    if not graph.has_edge(a, u, graph.bidirected_edge_name) and not graph.has_edge(
        u, a, graph.directed_edge_name
    ):
        return found_discriminating_path, disc_path, explored_nodes

    # now add 'a' to the queue and begin exploring
    # adjacent nodes that are connected with bidirected edges
    path = deque([a])
    while len(path) != 0:
        this_node = path.popleft()

        # check distance criterion to prevent checking very long paths
        distance += 1
        if distance > 0 and distance > max_path_length:
            logger.warning(
                f"Did not finish checking discriminating path in {graph} because the path "
                f"length exceeded {max_path_length}."
            )
            return found_discriminating_path, disc_path, explored_nodes

        # now we check all neighbors to this_node that are pointing to it
        # either with a direct edge, or a bidirected edge
        node_iterator = chain(graph.possible_parents(this_node), graph.parents(this_node))
        node_iterator = chain(node_iterator, graph.sub_bidirected_graph().neighbors(this_node))

        # 'next_node' is either a parent, possible parent, or in a bidrected
        # edge with 'this_node'.
        # 'this_node' is a definite collider since there was
        # confirmed an arrow pointing towards 'this_node'
        # and 'next_node' is connected to it via a bidirected arrow.
        for next_node in node_iterator:
            # if we have already explored this neighbor, then it is
            # already along the potentially discriminating path
            if next_node in explored_nodes:
                continue

            # keep track of explored_nodes
            explored_nodes.add(next_node)

            # Check if 'next_node' is now the end of the discriminating path.
            # Note we now have 3 edges in the path by construction.
            if c not in graph.neighbors(next_node) and next_node != c:
                logger.info(f"Reached the end of the discriminating path with {next_node}.")
                explored_nodes.add(next_node)
                descendant_nodes[next_node] = this_node
                found_discriminating_path = True
                break

            # If we didn't reach the end of the discriminating path,
            # then we can add 'next_node' to the path. This only occurs
            # if 'next_node' is a valid new entry, which requires it
            # to be a part of the parents of 'c'.
            if next_node in cparents and graph.has_edge(
                this_node, next_node, graph.bidirected_edge_name
            ):
                # check that the next_node is a possible collider with at least
                # this_node -> next_node
                # since it is a parent, we can now add it to the path queue
                path.append(next_node)
                descendant_nodes[next_node] = this_node
                explored_nodes.add(next_node)

    # return the actual discriminating path
    if found_discriminating_path:
        disc_path = deque([])  # type: ignore
        disc_path.append(next_node)
        while disc_path[-1] != c:
            disc_path.append(descendant_nodes[disc_path[-1]])

    return found_discriminating_path, disc_path, explored_nodes


def uncovered_pd_path(
    graph: PAG,
    u: Node,
    c: Node,
    max_path_length: Optional[int] = None,
    first_node: Optional[Node] = None,
    second_node: Optional[Node] = None,
    force_circle: bool = False,
    forbid_node: Optional[Node] = None,
) -> Tuple[List[Node], bool]:
    """Compute uncovered potentially directed (pd) paths from u to c.


    In a pd path, the edge between V(i) and V(i+1) is not an arrowhead into V(i)
    or a tail from V(i+1). An intuitive explanation given in :footcite:`Zhang2008`
    notes that a pd path could be oriented into a directed path by changing circles
    into tails or arrowheads.

    In addition, the path is uncovered, meaning every node beside the endpoints are unshielded,
    meaning V(i-1) and V(i+1) are not adjacent.

    A special case of a uncovered pd path is an uncovered circle path, which appears as u o-o ... o-o c.

    Parameters
    ----------
    graph : PAG
        PAG to orient.
    u : node
        A node in the graph to start the uncovered path.
    c : node
        A node in the graph.
    max_path_length : optional, int
        The maximum distance to check in the graph. By default None, which sets
        it to 1000.
    first_node : node, optional
        The node previous to 'u'. If it is before 'u', then we will check
        that 'u' is unshielded. If it is not passed, then 'u' is considered
        the first node in the path and hence does not need to be unshielded.
        Both 'first_node' and 'second_node' cannot be passed.
    second_node : node, optional
        The node after 'u' that the path must traverse. Both 'first_node'
        and 'second_node' cannot be passed.
    force_circle: bool
        Whether to search for only circle paths (u o-o ... o-o c) or all potentially directed paths.
        By default False, which searches for all potentially directed paths.
    forbid_node: node, optional
        A node after 'u' which is forbidden to immediately traverse when searching for a path.

    Notes
    -----
    The definition of an uncovered pd path is taken from :footcite:`Zhang2008`.

    Typically uncovered potentially directed paths are defined by two nodes. However,
    in one use case within the FCI algorithm, it is defined relative
    to an adjacent third node that comes before 'u'.

    References
    ----------
    .. footbibliography::
    """
    if first_node is not None and second_node is not None:
        raise RuntimeError(
            "Both first and second node cannot be set. Only set one of them. "
            "Read the docstring for more information."
        )

    if (
        any(node not in graph for node in (u, c))
        or (first_node is not None and first_node not in graph)
        or (second_node is not None and second_node not in graph)
    ):
        raise RuntimeError("Some nodes are not in graph... Double check function arguments.")

    if max_path_length is None:
        max_path_length = 1000

    explored_nodes: Set[Node] = set()
    found_uncovered_pd_path = False
    uncov_pd_path: List[Node] = []

    # keep track of the distance searched
    distance = 0
    start_node = u

    # keep track of the previous nodes, i.e. to build a path
    # from node (key) to its child along the path (value)
    descendant_nodes = dict()
    if first_node is not None:
        descendant_nodes[u] = first_node
    if second_node is not None:
        descendant_nodes[second_node] = u

    # keep track of paths of certain nodes that were already explored
    # start off with the valid triple <a, u, c>
    # - u is adjacent to c
    # - u has an arrow pointing to a
    # - TBD a is a definite collider
    # - TBD endpoint is not adjacent to c
    explored_nodes.add(u)
    if first_node is not None:
        explored_nodes.add(first_node)
    if second_node is not None:
        explored_nodes.add(second_node)

        # we now want to start on the second_node
        start_node = second_node

    # now add 'a' to the queue and begin exploring
    # adjacent nodes that are connected with bidirected edges
    path = deque([start_node])
    while len(path) != 0:
        this_node = path.popleft()
        prev_node = descendant_nodes.get(this_node)

        # check distance criterion to prevent checking very long paths
        distance += 1
        if distance > 0 and distance > max_path_length:
            logger.warning(
                f"Did not finish checking discriminating path in {graph} because the path "
                f"length exceeded {max_path_length}."
            )
            return uncov_pd_path, found_uncovered_pd_path

        # get all adjacent nodes to 'this_node'
        for next_node in graph.neighbors(this_node):
            if this_node == start_node:
                if forbid_node is not None:
                    if next_node == forbid_node:
                        continue

            # if we have already explored this neighbor, then ignore
            if next_node in explored_nodes:
                continue

            # now check that the next_node is uncovered by comparing
            # with the previous node, because the triple is shielded
            if prev_node is not None and next_node in graph.neighbors(prev_node):
                continue

            # now check that the triple is potentially directed, else
            # we skip this node
            condition = graph.has_edge(this_node, next_node, graph.circle_edge_name)
            if not force_circle:
                # If we do not restrict to circle paths then directed edges are also OK
                condition = condition or graph.has_edge(
                    this_node, next_node, graph.directed_edge_name
                )
            if not condition:
                continue

            # now this next node is potentially directed, does not
            # form a shielded triple, so we add it to the path
            explored_nodes.add(next_node)
            descendant_nodes[next_node] = this_node

            # if we have reached our end node, then we have found an
            # uncovered possibly-directed path
            if next_node == c:
                logger.info(f"Reached the end of the uncovered pd path with {next_node}.")
                found_uncovered_pd_path = True
                break

            path.append(next_node)

    # return the actual uncovered pd path
    if first_node is None:
        first_node = u

    if found_uncovered_pd_path:
        uncov_pd_path_: deque = deque([])
        uncov_pd_path_.appendleft(c)
        while uncov_pd_path_[0] != first_node:
            uncov_pd_path_.appendleft(descendant_nodes[uncov_pd_path_[0]])
        uncov_pd_path = list(uncov_pd_path_)
    return uncov_pd_path, found_uncovered_pd_path


def pds(
    graph: PAG, node_x: Node, node_y: Node = None, max_path_length: Optional[int] = None
) -> Set[Node]:
    """Find all PDS sets between node_x and node_y.

    Parameters
    ----------
    graph : PAG
        The graph.
    node_x : node
        The node 'x'.
    node_y : node
        The node 'y'.
    max_path_length : optional, int
        The maximum length of a path to search on. By default None, which sets
        it to 1000.

    Returns
    -------
    dsep : set
        The possibly d-separating set between node_x and node_y.

    Notes
    -----
    Possibly d-separating (PDS) sets are nodes V, along an adjacency paths from
    'node_x' to some 'V', which has the following characteristics for every
    subpath triple <X, Y, Z> on the path:

    - Y is a collider, or
    - Y is a triangle (i.e. X, Y and Z form a complete subgraph)

    If the path meets these characteristics, then 'V' is in the PDS set.

    If Y is a triangle, then it will be uncertain with circular edges
    due to the fact that it is a shielded triple, not allowing us to infer
    that it is a collider. These are defined in :footcite:`Colombo2012`.

    References
    ----------
    .. footbibliography::
    """
    if max_path_length is None:
        max_path_length = 1000

    distance = 0
    edge = None
    # possibly d-sep set
    dsep: Set[Node] = set()

    # a queue to
    q: deque = deque()
    seen_edges = set()
    node_list: Optional[List[Node]] = []

    # keep track of previous nodes along the path for every node
    # along a path
    previous = {node_x: None}

    # get the adjacency graph to perform path searches over
    adj_graph = graph.to_undirected()

    if node_y is not None:
        # edge case: check that there exists paths between node_x
        # and node_y
        if not nx.has_path(adj_graph, node_x, node_y):
            return dsep

    # get a list of all neighbors of node_x that is not y
    # and add these as candidates to explore a path
    # and also add them to the possibly d-separating set
    for node_v in graph.neighbors(node_x):
        # ngbhr cannot be endpoint
        if node_v == node_y:
            continue

        if node_y is not None:
            # used for RFCI
            # check that node_b is connected to the endpoint if
            # the endpoint is passed
            if not nx.has_path(adj_graph, node_v, node_y):
                continue

        # form edge as a tuple
        edge = (node_x, node_v)

        # this path from node_x - node_v is a candidate path
        # that will have a possibly d-separating set
        q.append(edge)

        # keep track of the edes
        seen_edges.add(edge)

        # all immediately adjacent nodes are part of the pdsep set
        dsep.add(node_v)

    while len(q) != 0:
        this_edge = q.popleft()
        prev_node, this_node = this_edge

        # if we get the previous edge, then increment the distance
        # and
        if this_edge == edge:
            edge = None
            distance += 1
            if distance > 0 and distance > max_path_length:
                break

        if node_y is not None:
            # check that node_b is connected to the endpoint if
            # the endpoint is passed
            if not nx.has_path(adj_graph, this_node, node_y):
                continue

        # now add this_node to the pdsep set, since we have
        # reached this node
        dsep.add(this_node)

        # now we want to check the subpath that is created
        # using the previous node, the current node and the next node
        for next_node in graph.neighbors(this_node):
            # check if 'node_c' in (prev_node, X, Y)
            if next_node in (prev_node, node_x, node_y):
                continue

            # get the previous nodes, and add the previous node
            # for this next node
            node_list = previous.get(next_node)
            if node_list is None:
                node_list = []
            node_list.append(this_node)

            # check that we have a definite collider
            # check the edge: prev_node - this_node
            # check the edge: this_node - next_node
            is_def_collider = is_definite_collider(graph, prev_node, this_node, next_node)

            # check that there is a triangle, meaning
            # prev_node is adjacent to next_node
            is_triangle = next_node in graph.neighbors(prev_node)

            # if we have a collider, or triangle, then this edge
            # is a candidate on a pdsep path
            if is_def_collider or is_triangle:
                next_edge = (prev_node, next_node)
                if next_edge in seen_edges:
                    continue

                seen_edges.add(next_edge)
                q.append(next_edge)
                if edge is None:
                    edge = next_edge

    return dsep


def pds_path(
    graph: PAG, node_x: Node, node_y: Node, max_path_length: Optional[int] = None
) -> Set[Node]:
    """Compute the possibly-d-separating set path.

    Returns the PDS_path set defined in definition 3.4 of :footcite:`Colombo2012`.

    Parameters
    ----------
    graph : PAG
        The graph.
    node_x : node
        The starting node.
    node_y : node
        The ending node
    max_path_length : int, optional
        The maximum length of a path to search on for PDS set, by default None, which
        sets it to 1000.

    Returns
    -------
    pds_path : set
        The set of nodes in the possibly d-separating path set.

    Notes
    -----
    This is a smaller subset compared to possibly-d-separating sets. It takes
    the PDS set and intersects it with the biconnected components of the adjacency
    graph that contains the edge (node_x, node_y).

    The current implementation calls `pds` and then restricts the nodes that it returns.
    """
    # get the adjacency graph to perform path searches over
    adj_graph = graph.to_undirected()

    # compute all biconnected componnets
    biconn_comp = nx.biconnected_component_edges(adj_graph)

    # compute the PDS set
    pds_set = pds(graph, node_x=node_x, node_y=node_y, max_path_length=max_path_length)

    # now we intersect the connected component that has the edge
    found_component: Set = set()
    for comp in biconn_comp:
        if (node_x, node_y) in comp or (node_y, node_x) in comp:
            # add all unique nodes in the biconnected component
            for (x, y) in comp:
                found_component.add(x)
                found_component.add(y)
            break

    # now intersect the pds set with the biconnected component with the edge between
    # 'x' and 'y'
    pds_path = pds_set.intersection(found_component)

    return pds_path


def pds_t(
    graph: StationaryTimeSeriesPAG,
    node_x: TsNode,
    node_y: TsNode,
    max_path_length: Optional[int] = None,
) -> Set:
    """Compute the possibly-d-separating set over time.

    Returns the 'pdst' set defined in :footcite:`Malinsky18a_svarfci`.

    Parameters
    ----------
    graph : StationaryTimeSeriesPAG
        The graph.
    node_x : node
        The starting node.
    node_y : node
        The ending node
    max_path_length : int, optional
        The maximum length of a path to search on for PDS set, by default None, which
        sets it to 1000.

    Returns
    -------
    pds_t_set : set
        The set of nodes in the possibly d-separating path set.

    Notes
    -----
    This is a smaller subset compared to possibly-d-separating sets.

    This consists of nodes, 'x', in the PDS set of (node_x, node_y), with the
    time-lag of 'x' being less than the max time-lag among node_x and and node_y.

    The current implementation calls `pds` and then restricts the nodes that it returns.
    """
    _check_ts_node(node_x)
    _check_ts_node(node_y)
    _, x_lag = node_x
    _, y_lag = node_y

    max_lag = max(np.abs(x_lag), np.abs(y_lag))

    # compute the PDS set
    pds_set = pds(
        graph, node_x=node_x, node_y=node_y, max_path_length=max_path_length
    )  # type: ignore
    pds_t_set = set()

    # only keep nodes with max-lag less than or equal to max(x_lag, y_lag)
    for node in pds_set:
        if np.abs(node[1]) <= max_lag:  # type: ignore
            pds_t_set.add(node)

    return pds_t_set


def pds_t_path(
    graph: StationaryTimeSeriesPAG,
    node_x: TsNode,
    node_y: TsNode,
    max_path_length: Optional[int] = None,
) -> Set:
    """Compute the possibly-d-separating path set over time.

    Returns the 'pdst_path' set defined in :footcite:`Malinsky18a_svarfci` with the
    additional restriction that any nodes must be on a path between the two endpoints.

    Parameters
    ----------
    graph : StationaryTimeSeriesPAG
        The graph.
    node_x : node
        The starting node.
    node_y : node
        The ending node
    max_path_length : int, optional
        The maximum length of a path to search on for PDS set, by default None, which
        sets it to 1000.

    Returns
    -------
    pds_t_set : set
        The set of nodes in the possibly d-separating path set.

    Notes
    -----
    This is a smaller subset compared to possibly-d-separating sets.

    This consists of nodes, 'x', in the PDS set of (node_x, node_y), with the
    time-lag of 'x' being less than the max time-lag among node_x and and node_y.

    The current implementation calls `pds` and then restricts the nodes that it returns.
    """
    _check_ts_node(node_x)
    _check_ts_node(node_y)
    _, x_lag = node_x
    _, y_lag = node_y

    max_lag = max(np.abs(x_lag), np.abs(y_lag))

    # compute the PDS set
    pds_set = pds_path(
        graph, node_x=node_x, node_y=node_y, max_path_length=max_path_length
    )  # type: ignore
    pds_t_set = set()

    # only keep nodes with max-lag less than or equal to max(x_lag, y_lag)
    for node in pds_set:
        if np.abs(node[1]) <= max_lag:  # type: ignore
            pds_t_set.add(node)

    return pds_t_set


def definite_m_separated(
    G,
    x,
    y,
    z,
    bidirected_edge_name="bidirected",
    directed_edge_name="directed",
    circle_edge_name="cirlcle",
):
    """Check definite m-separation among 'x' and 'y' given 'z' in partial ancestral graph G.

    A partial ancestral graph (PAG) is defined with directed edges (``->``), bidirected edges
    (``<->``), and circle-endpoint edges (``o-*``, where the ``*`` for example can mean an
    arrowhead from a directed edge).

    This algorithm implements the definite m-separation check, which checks for the absence of
    possibly m-connecting paths between 'x' and 'y' given 'z'.

    This algorithm first obtains the ancestral subgraph of x | y | z which only requires knowledge
    of the directed edges. Then, all outgoing directed edges from nodes in z are deleted. After
    that, an undirected graph composed from the directed and bidirected edges amongst the
    remaining nodes is created. Then, x is independent of y given z if x is disconnected from y
    in this new graph.

    Parameters
    ----------
    G : mixed-edge-graph
        Mixed edge causal graph.
    x : set
        First set of nodes in ``G``.
    y : set
        Second set of nodes in ``G``.
    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    Returns
    -------
    b : bool
        A boolean that is true if ``x`` is definite m-separated from ``y`` given ``z`` in ``G``.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    d_separated
    m_separated
    PAG

    Notes
    -----
    There is no known optimal algorithm for checking definite m-separation to our knowledge, so
    the algorithm proceeds by enumerating paths between 'x' and 'y'. This first checks the
    subgraph comprised of only circle edges. If there is a path
    """
    if not isinstance(G, PAG):
        raise ValueError("Definite m-separated is only defined for a PAG.")

    # this proceeds by first removing unnecessary nodes


def _check_ts_node(node):
    if not isinstance(node, tuple) or len(node) != 2:
        raise ValueError(
            f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
            f"You passed in {node}."
        )
    if node[1] > 0:
        raise ValueError(f"All lag points should be 0, or less. You passed in {node}.")
