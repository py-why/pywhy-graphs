import networkx as nx


def is_clique(G, nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n * (n - 1) / 2


def order_edges(G):
    pass


def label_edges(G):
    pass


def cpdag_to_pdag(G):
    """Convert a CPDAG to

    Parameters
    ----------
    G : _type_
        _description_
    """
    pass


def pdag_to_dag(G):
    """Compute consistent extension of given PDAG resulting in a DAG.

    Implements the algorithm described in Figure 11 of :footcite:`chickering2002learning`.

    Parameters
    ----------
    G : CPDAG
        A partially directed acyclic graph.

    Returns
    -------
    DAG
        A directed acyclic graph.

    References
    ----------
    .. footbibliography::
    """
    if set(["directed", "undirected"]) != set(G.edge_types):
        raise ValueError("Only directed and undirected edges are allowed in a CPDAG")

    dir_G: nx.DiGraph = G.get_graphs(edge_type="directed")
    undir_G: nx.Graph = G.get_graphs(edge_type="undirected")
    full_undir_G: nx.Graph = G.to_undirected()
    nodes = set(dir_G.nodes)
    found = False

    while nodes:
        found = False
        idx = 0

        # select a node, x, which:
        # 1. has no outgoing edges
        # 2. all undirected neighbors are adjacent to all its adjacent nodes
        while not found and idx < len(nodes):
            # check that there are no outgoing edges for said node
            node_is_sink = dir_G.out_degree(nodes[idx]) == 0

            if not node_is_sink:
                idx += 1
                continue

            # since there are no outgoing edges, all directed adjacencies are parent nodes
            # now check that all undirected neighbors are adjacent to all its adjacent nodes
            undir_nbrs = undir_G.neighbors(nodes[idx])
            parents = dir_G.predecessors(nodes[idx])
            undir_nbrs_and_parents = set(undir_nbrs).union(set(parents))
            nearby_is_clique = is_clique(full_undir_G, undir_nbrs_and_parents)
            idx += 1

            if nearby_is_clique:
                found = True

                # now, we orient all undirected edges between x and its neighbors
                # such that ``nbr -> x``
                for nbr in undir_nbrs:
                    dir_G.add_edge(nbr, nodes[idx], edge_type="directed")

                # remove x from the "graph"
                nodes.remove(nodes[idx])
    if not found:
        raise ValueError("No consistent extension found")
    return dir_G


def dag_to_cpdag(G):
    """Convert a DAG to a CPDAG.

    Parameters
    ----------
    G : _type_
        _description_
    """
    pass
