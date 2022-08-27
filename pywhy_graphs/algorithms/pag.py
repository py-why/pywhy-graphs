import logging
from collections import deque
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from pywhy_graphs import PAG
from pywhy_graphs.typing import Node

logger = logging.getLogger()


def discriminating_path(graph: PAG, u: Node, a: Node, c: Node, max_path_length: int = np.inf):
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
    max_path_length : int
        The maximum distance to check in the graph.

    Returns
    -------
    explored_nodes : dict
        A hash map of explored nodes.
    found_discriminating_path : bool
        Whether or not a discriminating path was found.
    disc_path : list
        The discriminating path starting from node c.
    """
    if max_path_length == np.inf:
        max_path_length = 1000

    explored_nodes: Dict[Node, None] = dict()
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
    explored_nodes[c] = None
    explored_nodes[u] = None
    explored_nodes[a] = None

    # a must be a parent of c
    if not graph.has_edge(a, c):
        return explored_nodes, found_discriminating_path, disc_path

    # a and u must be connected by a bidirected edge, or with an edge towards a
    # for a to be a definite collider
    if not graph.has_bidirected_edge(a, u) and not graph.has_edge(u, a):
        return explored_nodes, found_discriminating_path, disc_path

    # now add 'a' to the queue and begin exploring
    # adjacent nodes that are connected with bidirected edges
    path = deque([a])
    while not len(path) == 0:
        this_node = path.popleft()

        # check distance criterion to prevent checking very long paths
        distance += 1
        if distance > 0 and distance > max_path_length:
            logger.warn(
                f"Did not finish checking discriminating path in {graph} because the path "
                f"length exceeded {max_path_length}."
            )
            return explored_nodes, found_discriminating_path, disc_path

        # now we check all neighbors to this_node that are pointing to it
        # either with a direct edge, or a bidirected edge
        node_iterator = chain(graph.possible_parents(this_node), graph.parents(this_node))
        if this_node in graph.c_component_graph:
            node_iterator = chain(node_iterator, graph.c_component_graph.neighbors(this_node))

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
            explored_nodes[next_node] = None

            # Check if 'next_node' is now the end of the discriminating path.
            # Note we now have 3 edges in the path by construction.
            if not graph.has_adjacency(next_node, c) and next_node != c:
                logger.info(f"Reached the end of the discriminating path with {next_node}.")
                explored_nodes[next_node] = None
                descendant_nodes[next_node] = this_node
                found_discriminating_path = True
                break

            # If we didn't reach the end of the discriminating path,
            # then we can add 'next_node' to the path. This only occurs
            # if 'next_node' is a valid new entry, which requires it
            # to be a part of the parents of 'c'.
            if next_node in cparents and graph.has_bidirected_edge(this_node, next_node):
                # check that the next_node is a possible collider with at least
                # this_node -> next_node
                # since it is a parent, we can now add it to the path queue
                path.append(next_node)
                descendant_nodes[next_node] = this_node
                explored_nodes[next_node] = None

    # return the actual discriminating path
    if found_discriminating_path:
        disc_path = deque([])  # type: ignore
        disc_path.append(next_node)
        while disc_path[-1] != c:
            disc_path.append(descendant_nodes[disc_path[-1]])

    return explored_nodes, found_discriminating_path, disc_path


def uncovered_pd_path(
    graph: PAG, u, c, max_path_length: int, first_node=None, second_node=None
) -> Tuple[List, bool]:
    """Compute uncovered potentially directed path from u to c.

    An uncovered pd path is one where: u o-> ... -> c. There are no
    bidirected arrows, bidirected circle arrows, or opposite arrows.
    In addition, every node beside the endpoints are unshielded,
    meaning V(i-1) and V(i+1) are not adjacent.

    Parameters
    ----------
    graph : ADMG
        PAG to orient.
    u : node
        A node in the graph to start the uncovered path.
    c : node
        A node in the graph.
    max_path_length : int
        The maximum distance to check in the graph.
    first_node : node, optional
        The node previous to 'u'. If it is before 'u', then we will check
        that 'u' is unshielded. If it is not passed, then 'u' is considered
        the first node in the path and hence does not need to be unshielded.
        Both 'first_node' and 'second_node' cannot be passed.
    second_node : node, optional
        The node after 'u' that the path must traverse. Both 'first_node'
        and 'second_node' cannot be passed.

    Notes
    -----
    Typically uncovered potentially directed paths are defined by two nodes. However,
    in its common use case within the FCI algorithm, it is usually defined relative
    to an adjacent third node that comes before 'u'.
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

    if max_path_length == np.inf:
        max_path_length = 1000

    explored_nodes: Dict[Node, None] = dict()
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
    explored_nodes[u] = None
    if first_node is not None:
        explored_nodes[first_node] = None
    if second_node is not None:
        explored_nodes[second_node] = None

        # we now want to start on the second_node
        start_node = second_node

    # now add 'a' to the queue and begin exploring
    # adjacent nodes that are connected with bidirected edges
    path = deque([start_node])
    while not len(path) == 0:
        this_node = path.popleft()
        prev_node = descendant_nodes.get(this_node)

        # check distance criterion to prevent checking very long paths
        distance += 1
        if distance > 0 and distance > max_path_length:
            logger.warn(
                f"Did not finish checking discriminating path in {graph} because the path "
                f"length exceeded {max_path_length}."
            )
            return uncov_pd_path, found_uncovered_pd_path

        # get all adjacent nodes to 'this_node'
        for next_node in graph.adjacencies(this_node):
            # if we have already explored this neighbor, then ignore
            if next_node in explored_nodes:
                continue

            # now check that the next_node is uncovered by comparing
            # with the previous node, because the triple is shielded
            if prev_node is not None:
                if graph.has_adjacency(prev_node, next_node):
                    continue

            # now check that the triple is potentially directed, else
            # we skip this node
            if not graph.has_edge(this_node, next_node):
                continue

            # now this next node is potentially directed, does not
            # form a shielded triple, so we add it to the path
            explored_nodes[next_node] = None
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
        uncov_pd_path = deque([])  # type: ignore
        uncov_pd_path.appendleft(c)  # type: ignore
        while uncov_pd_path[0] != first_node:
            uncov_pd_path.appendleft(descendant_nodes[uncov_pd_path[0]])  # type: ignore
        uncov_pd_path = list(uncov_pd_path)
    return uncov_pd_path, found_uncovered_pd_path


def possibly_d_sep_sets(graph: PAG, node_x, node_y=None, max_path_length: int = np.inf) -> Set:
    """Find all PDS sets between node_x and node_y.

    Possibly d-separting (PDS) sets are adjacency paths from 'node_x' to
    some node 'V', which has the following characteristics for every
    subpath triple <X, Y, Z> on the path:

    - Y is a collider, or
    - Y is a triangle (i.e. X, Y and Z form a complete subgraph)

    If Y is a triangle, then it will be uncertain with circular edges
    due to the fact that it is a shielded triple, not allowing us to infer
    that it is a collider. These are defined in :footcite:`Colombo2012`.

    Parameters
    ----------
    graph : PAG
        _description_
    node_x : node
        The node 'x'.
    node_y : node
        The node 'y'.
    max_path_length : int
        The maximum length of a path to search on.

    Returns
    -------
    dsep : set
        The possibly d-separating set between node_x and node_y.
    """
    if max_path_length == np.inf:
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
    adj_graph = graph.to_adjacency_graph()

    if node_y is not None:
        # edge case: check that there exists paths between node_x
        # and node_y
        if not nx.has_path(adj_graph, node_x, node_y):
            return dsep

    # get a list of all neighbors of node_x that is not y
    # and add these as candidates to explore a path
    # and also add them to the possibly d-separating set
    for node_v in graph.adjacencies(node_x):
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
        for next_node in graph.adjacencies(this_node):
            # check if 'node_c' in (X, Y, prev_node)
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
            is_def_collider = graph.is_def_collider(prev_node, this_node, next_node)

            # check that there is a triangle, meaning
            # prev_node is adjacent to next_node
            is_triangle = graph.has_adjacency(prev_node, next_node)

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


def pds_path(graph: PAG, node_x, node_y, max_path_length: int = np.inf) -> Set:
    """Compute the possibly-d-separating set path.

    Returns the PDS_path set defined in definition 3.4 of :footcite:`Colombo2012`.

    Parameters
    ----------
    graph : PAG
        _description_
    node_x : node
        The starting node.
    node_y : node
        The ending node
    max_path_length : int, optional
        The maximum length of a path to search on for PDS set, by default np.inf.

    Notes
    -----
    This is a smaller subset compared to possibly-d-separating sets. It takes
    the PDS set and intersects it with the biconnected components of the adjacency
    graph that contains the edge (node_x, node_y).
    """
    # get the adjacency graph to perform path searches over
    adj_graph = graph.to_adjacency_graph()

    # compute all biconnected componnets
    biconn_comp = nx.biconnected_component_edges(adj_graph)

    # compute the PDS set
    pds_set = possibly_d_sep_sets(
        graph, node_x=node_x, node_y=node_y, max_path_length=max_path_length
    )

    # now we intersect
    found_component: Set = set()
    for comp in biconn_comp:
        if (node_x, node_y) in biconn_comp or (node_y, node_x) in biconn_comp:
            # add all unique nodes in the biconnected component
            for (x, y) in comp:
                found_component.add(x)
                found_component.add(y)
            break

    # now intersect the pds set with the biconnected component with the edge between
    # 'x' and 'y'
    pds_path = pds_set.intersection(found_component)

    return pds_path
