from enum import Enum

import networkx as nx

import pywhy_graphs as pg

__all__ = ["pdag_to_dag", "dag_to_cpdag", "pdag_to_cpdag", "order_edges", "label_edges"]


class EDGELABELS(Enum):
    """Edge labels for a CPDAG."""

    COMPELLED = "compelled"
    REVERSIBLE = "reversible"
    UNKNOWN = "unknown"


def is_clique(G, nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n * (n - 1) / 2


def order_edges(G: nx.DiGraph):
    """Find total ordering of the edges of DAG G.

    A total ordering is a topological sorting of the nodes, and then
    ordering all possible edges according to Algorithm 4 in
    :footcite:`chickering2002learning`.

    Parameters
    ----------
    G : DAG
        A directed acyclic graph.

    Returns
    -------
    list
        A list of edges in the DAG.

    References
    ----------
    .. footbibliography::
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a directed acyclic graph")
    nx.set_edge_attributes(G, None, "order")
    ordered_nodes = list(nx.topological_sort(G))

    idx = 0

    while any([G[u][v]["order"] is None for u, v in G.edges]):
        # get all edges that are still not ordered
        unordered_edges = [(u, v) for u, v in G.edges if G[u][v]["order"] is None]

        # get the lowest order unlabeled edge's destination node
        y = sorted(unordered_edges, key=lambda x: ordered_nodes.index(x[1]))[-1][-1]

        # find the highest order node such that x -> y is not ordered
        unlabeled_y_parent_edges = [u for u in G.predecessors(y) if G[u][y]["order"] is None]
        x = sorted(unlabeled_y_parent_edges, key=lambda x: ordered_nodes.index(x))[0]

        # label the edge order
        G[x][y]["order"] = idx
        idx += 1

    return G


def label_edges(G: nx.DiGraph):
    """Label compelled and reversible edges of a DAG G.

    Label the edges of a DAG G as either compelled or reversible. Compelled
    edges are edges that are compelled to be directed in a consistent
    extension of G. Reversible edges are edges that are not required
    to be directed in a consistent extension of G. For full details,
    see Algorithm 5 in :footcite:`chickering2002learning`.

    Parameters
    ----------
    G : DAG
        The directed acyclic graph to label.

    Returns
    -------
    DAG
        The labelled DAG with edge attribute ``"label"`` as either
        ``"compelled"`` or ``"reversible"``.

    References
    ----------
    .. footbibliography::
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a directed acyclic graph")
    if not all([G[u][v].get("order") is not None for u, v in G.edges]):
        raise ValueError("G must have all edges ordered via the `order` attribute")

    nx.set_edge_attributes(G, EDGELABELS.UNKNOWN, "label")

    while any([edge[-1] == EDGELABELS.UNKNOWN for edge in G.edges.data("label")]):
        # find the lowest order edge with an unknown label
        unknown_edges = [
            (src, target)
            for src, target, label in G.edges.data("label")
            if label == EDGELABELS.UNKNOWN
        ]
        unknown_edges.sort(key=lambda edge: G.edges[edge]["order"])
        x, y = unknown_edges[-1]

        # now find every edge w -> x that is labeled as compelled
        w_nodes = [w for w in G.predecessors(x) if G[w][x]["label"] == EDGELABELS.COMPELLED]
        continue_while_loop = False
        for node in w_nodes:
            # For all compelled edges w -> x, if there is no edge w -> y,
            # we can label the edge x -> y as compelled
            if not G.has_edge(node, y):
                for src, target in G.in_edges(y):
                    G[src][target]["label"] = EDGELABELS.COMPELLED

                # now, we start over at the beginning of the while loop
                continue_while_loop = True
                break
            else:
                # w -> y is compelled, since there is an edge w -> x that is compelled
                # so w is a confounder
                G[node][y]["label"] = EDGELABELS.COMPELLED

        if continue_while_loop:
            continue

        # now, we check if there an edge z -> y such that:
        # 1. z != x
        # 2. z is not a parent of x
        # If so, then label all unknown edges into y (including x -> y)
        # as compelled
        # otherwise, label all unknown edges with reversible label
        z_exists = len([z for z in G.predecessors(y) if z != x and not G.has_edge(z, x)])
        for src, target in G.in_edges(y):
            if G[src][target]["label"] == EDGELABELS.UNKNOWN:
                if z_exists:
                    G[src][target]["label"] = EDGELABELS.COMPELLED
                else:
                    G[src][target]["label"] = EDGELABELS.REVERSIBLE
    return G


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

    G = G.copy()
    dir_G: nx.DiGraph = G.get_graphs(edge_type="directed")
    undir_G: nx.Graph = G.get_graphs(edge_type="undirected")
    full_undir_G: nx.Graph = G.to_undirected()

    # initialize a DAG for the consistent extension
    dag = nx.DiGraph(dir_G)

    nodes_memo = {node: None for node in G.nodes}
    found = False

    while len(nodes_memo) > 0:
        found = False
        idx = 0

        nodes = list(nodes_memo.keys())

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
            undir_nbrs = list(undir_G.neighbors(nodes[idx]))
            nearby_is_clique = False
            if len(undir_nbrs) != 0:
                parents = dir_G.predecessors(nodes[idx])
                # adj = full_undir_G.neighbors(nodes[idx])
                undir_nbrs_and_parents = set(undir_nbrs).union(set(parents))
                nearby_is_clique = is_clique(full_undir_G, undir_nbrs_and_parents)

            if len(undir_nbrs) == 0 or nearby_is_clique:
                found = True

                # now, we orient all undirected edges between x and its neighbors
                # such that ``nbr -> x``
                for nbr in undir_nbrs:
                    dag.add_edge(nbr, nodes[idx], edge_type="directed")

                # remove x from the "graph" and memoization
                del nodes_memo[nodes[idx]]
                dir_G.remove_node(nodes[idx])
                undir_G.remove_node(nodes[idx])
                full_undir_G.remove_node(nodes[idx])
            else:
                idx += 1

        # if no node satisfies condition 1 and 2, then the PDAG does not
        # admit a consistent extension
        if not found:
            print(nodes_memo)
            raise ValueError(f"No consistent extension found for PDAG: {G}, {G.edges()}")
    return dag


def dag_to_cpdag(G):
    """Convert a DAG to a CPDAG.

    Creates a CPDAG from a DAG.

    Parameters
    ----------
    G : DAG
        Directed acyclic graph.
    """
    G = order_edges(G)
    G = label_edges(G)

    # now construct CPDAG
    cpdag = pg.CPDAG()

    # for all compelled edges, add a directed edge
    compelled_edges = [
        (u, v) for u, v, label in G.edges.data("label") if label == EDGELABELS.COMPELLED
    ]
    cpdag.add_edges_from(compelled_edges, edge_type="directed")

    # for all reversible edges, add an undirected edge
    reversible_edges = [
        (u, v) for u, v, label in G.edges.data("label") if label == EDGELABELS.REVERSIBLE
    ]
    cpdag.add_edges_from(reversible_edges, edge_type="undirected")

    return cpdag


def pdag_to_cpdag(G):
    """Convert a PDAG to a CPDAG.

    Parameters
    ----------
    G : PDAG
        A partially directed acyclic graph that is not completed.

    Returns
    -------
    CPDAG
        A completed partially directed acyclic graph.
    """
    dag = pdag_to_dag(G)

    return dag_to_cpdag(dag)
