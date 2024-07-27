import networkx as nx

from ..config import EdgeType
from ..typing import Node

__all__ = [
    "is_semi_directed_path",
    "all_semi_directed_paths",
]


def _empty_generator():
    yield from ()


def is_semi_directed_path(G, nodes):
    """Returns True if and only if `nodes` form a semi-directed path in `G`.

    A *semi-directed path* in a graph is a nonempty sequence of nodes in which
    no node appears more than once in the sequence, each adjacent
    pair of nodes in the sequence is adjacent in the graph and where each
    pair of adjacent nodes does not contain a directed endpoint in the direction
    towards the start of the sequence.

    That is ``(a -> b o-> c <-> d -> e)`` is not a semi-directed path from ``a`` to ``e``
    because ``d *-> c`` is a directed endpoint in the direction towards ``a``.

    Parameters
    ----------
    G : graph
        A mixed-edge graph.
    nodes : list
        A list of one or more nodes in the graph `G`.

    Returns
    -------
    bool
        Whether the given list of nodes represents a semi-directed path in `G`.

    Notes
    -----
    This function is very similar to networkx's
    :func:`networkx.algorithms.simple_paths.is_simple_path` function.
    """
    # The empty list is not a valid path. Could also return
    # NetworkXPointlessConcept here.
    if len(nodes) == 0:
        return False

    # If the list is a single node, just check that the node is actually
    # in the graph.
    if len(nodes) == 1:
        return nodes[0] in G

    # check that all nodes in the list are in the graph, if at least one
    # is not in the graph, then this is not a semi-directed path
    if not all(n in G for n in nodes):
        return False

    # If the list contains repeated nodes, then it's not a semi-directed path
    if len(set(nodes)) != len(nodes):
        return False

    # Test that each adjacent pair of nodes is adjacent and that there
    # is no directed endpoint towards the beginning of the sequence.
    for idx in range(len(nodes) - 1):
        u, v = nodes[idx], nodes[idx + 1]
        if G.has_edge(v, u, EdgeType.DIRECTED.value) or G.has_edge(v, u, EdgeType.BIDIRECTED.value):
            return False
        elif not G.has_edge(u, v):
            return False
    return True


def all_semi_directed_paths(G, source: Node, target: Node, cutoff: int = None):
    """Generate all semi-directed paths from source to target in G.

    A semi-directed path is a path from ``source`` to ``target`` in that
    no end-point is directed from ``target`` to ``source``. I.e.
    ``target *-> source`` does not exist.

    Parameters
    ----------
    G : Graph
        The graph.
    source : Node
        The source node.
    target : Node
        The target node.
    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Notes
    -----
    This algorithm is very similar to networkx's
    :func:`networkx.algorithms.simple_paths.all_simple_paths` function.

    This algorithm uses a modified depth-first search to generate the
    paths [1]_.  A single path can be found in $O(V+E)$ time but the
    number of semi-directed paths in a graph can be very large, e.g. $O(n!)$ in
    the complete graph of order $n$.

    This function does not check that a path exists between `source` and
    `target`. For large graphs, this may result in very long runtimes.
    Consider using `has_path` to check that a path exists between `source` and
    `target` before calling this function on large graphs.

    References
    ----------
    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",
       Addison Wesley Professional, 3rd ed., 2001.
    """
    if source not in G:
        raise nx.NodeNotFound("source node %s not in graph" % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)  # type: ignore
        except TypeError:
            raise nx.NodeNotFound("target node %s not in graph" % target)
    if source in targets:
        return _empty_generator()
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return _empty_generator()
    if cutoff is None:
        cutoff = len(G) - 1

    return _all_semi_directed_paths_graph(G, source, targets, cutoff)


def _all_semi_directed_paths_graph(
    G, source, targets, cutoff, directed_edge_name="directed", bidirected_edge_name="bidirected"
):
    """See networkx's all_simple_paths function.

    This performs a depth-first search for all semi-directed paths from source to target.
    """
    # memoize each node that was already visited
    visited = {source: True}

    # iterate over neighbors of source
    stack = [iter(G.neighbors(source))]

    # if source has no neighbors, then prev_nodes should be None
    prev_nodes = [source]

    while stack:
        # get the iterator through nbrs for the current node
        nbrs = stack[-1]
        prev_node = prev_nodes[-1]
        nbr = next(nbrs, None)

        # The first condition guarantees that there is not a directed endpoint
        # along the path from source to target that points towards source.
        if (
            G.has_edge(nbr, prev_node, directed_edge_name)
            or G.has_edge(nbr, prev_node, bidirected_edge_name)
        ) and nbr not in visited:
            # If we've found a directed edge from child to prev_node,
            # that we haven't visited, then we don't need to continue down this path
            continue
        elif nbr is None:
            # once all children are visited, pop the stack
            # and remove the child from the visited set
            stack.pop()
            visited.popitem()
            prev_nodes.pop()
        elif len(visited) < cutoff:
            if nbr in visited:
                continue
            if nbr in targets:
                # we've found a path to a target
                yield list(visited) + [nbr]
            visited[nbr] = True
            if targets - set(visited.keys()):  # expand stack until find all targets
                stack.append(iter(G.neighbors(nbr)))
                prev_nodes.append(nbr)
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            for target in (targets & (set(nbrs) | {nbr})) - set(visited.keys()):
                yield list(visited) + [target]
            stack.pop()
            visited.popitem()
            prev_nodes.pop()
