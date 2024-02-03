import logging
from collections import deque
from itertools import chain, combinations, permutations
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from dodiscover import FCI, make_context
from dodiscover.ci import Oracle
from dodiscover.constraint.utils import dummy_sample

from pywhy_graphs import ADMG, CPDAG, PAG, StationaryTimeSeriesPAG
from pywhy_graphs.algorithms.generic import (
    has_adc,
    inducing_path,
    single_source_shortest_mixed_path,
    valid_mag,
)
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
    "is_definite_noncollider",
    "pag_to_mag",
    "legal_pag",
    "equivalent_pag",
    "valid_pag",
]


def _possibly_directed(G: PAG, i: Node, j: Node, reverse: bool = False):
    """Check that edge is possibly directed.

    A possibly directed edge is one of the form:
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
    # everything else is valid; i.e. i -- j, or i o-o j
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

    Definite noncolliders have the form:

    - node1 *-* node2 -> node3, or
    - node1 <- node2 *-* node3, or
    - node1 *-o node2 o-* node3 with node1 and node3 non-adjacent

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
        Whether or not the path is a definite non-collider. If it is not a definite non-collider,
        then it may be a definite collider, or uncertain.
    """
    if G.has_edge(node1, node2, G.directed_edge_name) or G.has_edge(
        node1, node2, G.bidirected_edge_name
    ):
        # node1 *-> node2 *-* node3
        # or node1 *-* node2 <-* node3
        if G.has_edge(node3, node2, G.directed_edge_name) or G.has_edge(
            node3, node2, G.bidirected_edge_name
        ):
            return False
    elif G.has_edge(node1, node2, G.circle_edge_name) and G.has_edge(
        node3, node2, G.circle_edge_name
    ):
        # node1 *-o node2 o-* node3
        if G.has_edge(node1, node3, "any") or G.has_edge(node3, node1, "any"):
            return False
    return True


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
    disc_path : list
        The discriminating path starting from node c.
    found_discriminating_path : bool
        Whether or not a discriminating path was found.
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

    A special case of a uncovered pd path is an uncovered circle path, which appears
    as u o-o ... o-o c.

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
        Whether to search for only circle paths (u o-o ... o-o c) or all
        potentially directed paths. By default False, which searches for all potentially
        directed paths.
    forbid_node: node, optional
        A node after 'u' which is forbidden to immediately traverse when searching for a path.

    Notes
    -----
    The definition of an uncovered pd path is taken from :footcite:`Zhang2008`.

    Typically uncovered potentially directed paths are defined by two nodes. However,
    in one use case within the FCI algorithm, it is defined relative
    to an adjacent third node that comes before 'u'.

    In certain cases (e.g. R5 of FCI) an uncovered pd path must be found between two variables,
    but these variables are already adjacent and connected by a trivial uncovered pd path.
    To prevent the function from returning this trivial path, the 'forbid_node' argument can be
    used.

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
            # check that this is the starting node and whether or not we are on a forbidden path
            if this_node == start_node and forbid_node is not None and next_node == forbid_node:
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
    graph: PAG, node_x: Node, node_y: Optional[Node] = None, max_path_length: Optional[int] = None
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
    that it is a collider. These are defined in :footcite:`Colombo2012`
    and :footcite:`Spirtes1993`.

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
            for x, y in comp:
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
    circle_edge_name="circle",
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


def _apply_meek_rules(graph: CPDAG) -> None:
    """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.
    These are known as the Meek rules :footcite:`Meek1995`. They are deterministic
    in the sense that they are logical characterizations of what edges must be
    present given the rest of the local graph structure.
    Parameters
    ----------
    graph : CPDAG
        A graph containing directed and undirected edges.
    """
    # For all the combination of nodes i and j, apply the following
    # rules.
    completed = False
    while not completed:  # type: ignore
        change_flag = False
        for i in graph.nodes:
            for j in graph.neighbors(i):
                if i == j:
                    continue
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                r1_add = _meek_rule1(graph, i, j)

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                r2_add = _meek_rule2(graph, i, j)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                r3_add = _meek_rule3(graph, i, j)

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                r4_add = _meek_rule4(graph, i, j)

                if any([r1_add, r2_add, r3_add, r4_add]) and not change_flag:
                    change_flag = True
        if not change_flag:
            completed = True
            break


def _meek_rule1(graph: CPDAG, i: str, j: str) -> bool:
    """Apply rule 1 of Meek's rules.
    Looks for i - j such that k -> i, such that (k,i,j)
    is an unshielded triple. Then can orient i - j as i -> j.
    """
    added_arrows = False

    # Check if i-j.
    if graph.has_edge(i, j, graph.undirected_edge_name):
        for k in graph.predecessors(i):
            # Skip if k and j are adjacent because then it is a
            # shielded triple
            if j in graph.neighbors(k):
                continue

            # check if the triple is in the graph's excluded triples
            if frozenset((k, i, j)) in graph.excluded_triples:
                continue

            # Make i-j into i->j
            graph.orient_uncertain_edge(i, j)

            added_arrows = True
            break
    return added_arrows


def _meek_rule2(graph: CPDAG, i: str, j: str) -> bool:
    """Apply rule 2 of Meek's rules.
    Check for i - j, and then looks for i -> k -> j
    triple, to orient i - j as i -> j.
    """
    added_arrows = False

    # Check if i-j.
    if graph.has_edge(i, j, graph.undirected_edge_name):
        # Find nodes k where k is i->k
        child_i = set()
        for k in graph.successors(i):
            if not graph.has_edge(k, i, graph.directed_edge_name):
                child_i.add(k)
        # Find nodes j where j is k->j.
        parent_j = set()
        for k in graph.predecessors(j):
            if not graph.has_edge(j, k, graph.directed_edge_name):
                parent_j.add(k)

        # Check if there is any node k where i->k->j.
        candidate_k = child_i.intersection(parent_j)
        # if the graph has excluded triples, we would check at this point
        if graph.excluded_triples:
            # check if the triple is in the graph's excluded triples
            # if so, remove them from the candidates
            for k in candidate_k:
                if frozenset((i, k, j)) in graph.excluded_triples:
                    candidate_k.remove(k)

        # if there are candidate 'k' nodes, then orient the edge accordingly
        if len(candidate_k) > 0:
            # Make i-j into i->j
            graph.orient_uncertain_edge(i, j)
            added_arrows = True
    return added_arrows


def _meek_rule3(graph: CPDAG, i: str, j: str) -> bool:
    """Apply rule 3 of Meek's rules.
    Check for i - j, and then looks for k -> j <- l
    collider, and i - k and i - l, then orient i -> j.
    """
    added_arrows = False

    # Check if i-j first
    if graph.has_edge(i, j, graph.undirected_edge_name):
        # For all the pairs of nodes adjacent to i,
        # look for (k, l), such that j -> l and k -> l
        for k, l_node in combinations(graph.neighbors(i), 2):
            # Skip if k and l are adjacent.
            if l_node in graph.neighbors(k):
                continue
            # Skip if not k->j.
            if graph.has_edge(j, k, graph.directed_edge_name) or (
                not graph.has_edge(k, j, graph.directed_edge_name)
            ):
                continue
            # Skip if not l->j.
            if graph.has_edge(j, l_node, graph.directed_edge_name) or (
                not graph.has_edge(l_node, j, graph.directed_edge_name)
            ):
                continue

            # check if the triple is inside graph's excluded triples
            if frozenset((l_node, i, k)) in graph.excluded_triples:
                continue

            # if i - k and i - l, then  at this point, we have a valid path
            # to orient
            if graph.has_edge(k, i, graph.undirected_edge_name) and graph.has_edge(
                l_node, i, graph.undirected_edge_name
            ):
                graph.orient_uncertain_edge(i, j)
                added_arrows = True
                break
    return added_arrows


def _meek_rule4(graph: CPDAG, i: str, j: str) -> bool:
    """Apply rule 4 of Meek's rules.
    Check for i - j, and then looks for i - k -> l -> j, to orient i - j as i -> j.
    """
    added_arrows = False

    # Check if i-j.
    if graph.has_edge(i, j, graph.undirected_edge_name):
        # Find nodes k where k is i-k
        adj_i = set()
        for k in graph.neighbors(i):
            if not graph.has_edge(k, i, graph.directed_edge_name):
                adj_i.add(k)

        # Find nodes l where j is l->j.
        parent_j = set()
        for k in graph.predecessors(j):
            if not graph.has_edge(j, k, graph.directed_edge_name):
                parent_j.add(k)

        # generate all permutations of sets containing neighbors of i and parents of j
        permut = permutations(adj_i, len(parent_j))
        unq = set()  # type: ignore
        for comb in permut:
            zipped = zip(comb, parent_j)
            unq.update(zipped)

        # check if these pairs have a directed edge between them and that k-j does not exist
        dedges = set(graph.directed_edges)
        undedges = set(graph.undirected_edges)
        candidate_k = set()
        for pair in unq:
            if pair in dedges:
                if (pair[0], j) not in undedges:
                    candidate_k.add(pair)

        # if there are candidate 'k->l' pairs, then orient the edge accordingly
        if len(candidate_k) > 0:
            # Make i-j into i->j
            # logger.info(f"R2: Removing edge {i}-{j} to form {i}->{j}.")
            graph.orient_uncertain_edge(i, j)
            added_arrows = True
    return added_arrows


def pag_to_mag(graph):
    """Sample a MAG from a PAG using Zhang's algorithm.

     Using the algorithm defined in Theorem 2 of :footcite:`Zhang2008`, which turns all
     o-> edges to -> and -o edges to ->, then it converts the graph into a DAG with
     no unshielded colliders using the meek rules.

    Parameters
    ----------
    G : Graph
        The PAG.

    Returns
    -------
    mag : Graph
        The MAG constructed from the PAG.
    """
    copy_graph = graph.copy()

    cedges = set(copy_graph.circle_edges)
    dedges = set(copy_graph.directed_edges)

    temp_cpdag = CPDAG()

    to_remove = []
    to_reorient = []
    to_add = []

    for u, v in cedges:
        if (v, u) in dedges:  # remove the circle end from a 'o-->' edge to make a '-->' edge
            to_remove.append((u, v))
        elif (v, u) not in cedges:  # reorient a '--o' edge to '-->'
            to_reorient.append((u, v))
        elif (v, u) in cedges and (
            v,
            u,
        ) not in to_add:  # add all 'o--o' edges to the cpdag
            to_add.append((u, v))
    for u, v in to_remove:
        copy_graph.remove_edge(u, v, copy_graph.circle_edge_name)
    for u, v in to_reorient:
        copy_graph.orient_uncertain_edge(u, v)
    for u, v in to_add:
        temp_cpdag.add_edge(v, u, temp_cpdag.undirected_edge_name)

    flag = True

    # convert the graph into a DAG with no unshielded colliders

    while flag:
        undedges = temp_cpdag.undirected_edges
        if len(undedges) != 0:
            for u, v in undedges:
                temp_cpdag.remove_edge(u, v, temp_cpdag.undirected_edge_name)
                temp_cpdag.add_edge(u, v, temp_cpdag.directed_edge_name)
                _apply_meek_rules(temp_cpdag)
                break
        else:
            flag = False

    mag = ADMG()  # provisional MAG

    # construct the final MAG

    for u, v in copy_graph.directed_edges:
        mag.add_edge(u, v, mag.directed_edge_name)

    for u, v in temp_cpdag.directed_edges:
        mag.add_edge(u, v, mag.directed_edge_name)

    return mag


def legal_pag(G: PAG, L: Optional[set] = None, S: Optional[set] = None):
    """Checks if the provided graph is a valid Partial ancestral graph (MAG).

    A valid PAG as defined in :footcite:`Zhang2008` is a mixed edge graph that
    has no directed or almost directed cycles and no inducing paths between
    any two non-adjacent pair of nodes.

    Parameters
    ----------
    G : Graph
        The graph.

    Returns
    -------
    is_valid : bool
        A boolean indicating whether the provided graph is a valid PAG or not.

    """

    if L is None:
        L = set()

    if S is None:
        S = set()

    directed_sub_graph = G.sub_directed_graph()

    all_nodes = set(G.nodes)

    # check if there are more than one edges b/w two nodes
    for node in all_nodes:
        nb = set(G.neighbors(node))
        for elem in nb:
            edge_data = G.get_edge_data(node, elem)
            if (edge_data["bidirected"] is not None) and (edge_data["directed"] is not None):
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

    # check if there are any inducing paths between non-adjacent nodes in the non-circle edge sub-graph

    dedges = list(G.edges()["directed"])
    # undedges = list(G.edges()["undirected"])
    biedges = list(G.edges()["bidirected"])

    temp_pag = PAG()

    temp_pag.add_edges_from(dedges, temp_pag.directed_edge_name)

    # can't remember why I only handle directed and bidirected edges

    # temp_pag.add_edges_from(undedges, temp_pag.undirected_edge_name)

    temp_pag.add_edges_from(biedges, temp_pag.bidirected_edge_name)

    all_nodes = set(temp_pag.nodes)

    for source in all_nodes:
        nb = set(temp_pag.neighbors(source))
        cur_set = all_nodes - nb
        cur_set.remove(source)
        for dest in cur_set:
            out = inducing_path(temp_pag, source, dest, L, S)
            if out[0] is True:
                return False

    return True


def mag_to_pag(G: PAG):
    """Converts the provided mag into a pag using the FCI algorithm.
    The FCI algorithms, as defined in :footcite:`Zhang2008` is a provably
    complete for learning all the tractable features of an MAG, thus
    producing a PAG.

    Parameters
    ----------
    G : MAG
        The MAG.

    Returns
    -------
    pag : PAG
        The PAG constructed from the MAG.
    """

    data = dummy_sample(G)
    oracle = Oracle(G)
    # ci_estimator = GSquareCITest(data_type="discrete")
    context = make_context().variables(data=data).build()
    fci = FCI(ci_estimator=oracle)
    fci.learn_graph(data, context)

    return fci.graph_


def equivalent_pag(G1: PAG, G2: PAG):
    """Check if the two provided PAGs are equivalent or not.
    This function compares the edges in both the graphs to determine
    equivalency.

    Parameters
    ----------
    G1 : PAG
        The first PAG.

    G2 : PAG
        The second PAG.

    Returns
    -------
    is_equivalent : bool
        A boolean indicating whether the two PAGs are equivalent or not.
    """

    g1_edges = G1.edges()
    g2_edges = G2.edges()

    if set(g1_edges["directed"]) != set(g2_edges["directed"]):
        return False

    elif set(g1_edges["undirected"]) != set(g2_edges["undirected"]):
        return False

    elif set(g1_edges["bidirected"]) != set(g2_edges["bidirected"]):
        return False

    elif set(g1_edges["circle"]) != set(g2_edges["circle"]):
        return False

    else:
        return True


def valid_pag(G: PAG):
    """Check if the provided PAG is valid or not.

    The function determines the validity by first converting the PAG
    into an MAG, then checking the validity of the said MAG. After the
    validity of the MAG has been established, the MAG is converted back
    into a PAG. Then the function checks to see if the original and the
    reconverted PAG are equivalent or not.

    Parameters
    ----------
    G : PAG
        The PAG.

    Returns
    -------
    is_valid : bool
        Boolean indicating whether the provided PAG is valid or not.
    """

    interim_bool = False

    # check if the graph is a vald PAG
    if not legal_pag(G):
        return False

    converted_mag = pag_to_mag(G)

    is_valid = valid_mag(converted_mag)

    if is_valid:
        interim_bool = True

    # convert the mag back to a pag
    rec_pag = mag_to_pag(converted_mag)

    # check if the converted pag is equivalent to the original

    if equivalent_pag(rec_pag, G):
        return interim_bool
    else:
        return False
