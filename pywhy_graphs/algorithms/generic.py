import networkx as nx

__all__ = ["single_source_shortest_mixed_path"]


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
