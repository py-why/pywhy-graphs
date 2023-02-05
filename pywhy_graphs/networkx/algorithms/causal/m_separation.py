from collections import deque
from copy import deepcopy

import networkx as nx
from networkx.utils import UnionFind

import pywhy_graphs.networkx as pywhy_nx

__all__ = ["m_separated"]


def m_separated(
    G,
    x,
    y,
    z,
    bidirected_edge_name="bidirected",
    directed_edge_name="directed",
    undirected_edge_name="undirected",
):
    """Check m-separation among 'x' and 'y' given 'z' in mixed-edge causal graph G, which may
    contain directed, bidirected, and undirected edges.

    This algorithm adapts the linear time algorithm presented in [1]_ currently implemented in
    `networkx.algorithms.d_separation` to work for mixed-edge causal graphs, using m-separation
    logic detailed in [2]_.

    This algorithm works by retaining select edges in each of the directed, bidirected, and
    undirected edge subgraphs (if supplied). Then, an undirected graph is created from
    the union of allsuch retained edges (without direction information), and then
    m-separation of x and y givenz is determined if x is disconnected from y in this graph.

    In the directed edge subgraph, nodes and associated edges are removed if they are childless
    and not in x | y | z; this process is repeated until no such nodes remain. Then, outgoing
    edges from z are removed. The remaining edges are retained.

    In the bidirected edge subgraph, nodes and associated edges are removed if they are not
    in x | y | z. The remaining edges are retained.

    In the undirected edge subgraph, all edges involving z are removed. The remaining edges are
    retained.

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
    directed_edge_name : str
        Name of the directed edge, default is directed.
    bidirected_edge_name : str
        Name of the bidirected edge, default is bidirected.
    undirected_edge_name : str
        Name of the undirected edge, default is undirected.

    Returns
    -------
    b : bool
        A boolean that is true if ``x`` is m-separated from ``y`` given ``z`` in ``G``.

    References
    ----------
    .. [1] Darwiche, A.  (2009).  Modeling and reasoning with Bayesian networks.
       Cambridge: Cambridge University Press.
    .. [2] Spirtes, P. and Richardson, T.S.. (1997). A Polynomial Time Algorithm
       for Determining DAG Equivalence in the Presence of Latent Variables and Selection
       Bias. Proceedings of the Sixth International Workshop on Artificial Intelligence and
       Statistics, in Proceedings of Machine Learning Research

    See Also
    --------
    networkx.algorithms.d_separation.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs and does
    not have an ``ADMG`` representation.
    """
    if not isinstance(G, pywhy_nx.MixedEdgeGraph):
        raise nx.NetworkXError(
            "m-separation should only be run on a MixedEdgeGraph. If "
            'you have a directed graph, use "d_separated" function instead.'
        )
    if not set(G.edge_types).issubset(
        {directed_edge_name, bidirected_edge_name, undirected_edge_name}
    ):
        raise nx.NetworkXError(
            f"m-separation only works on graphs with directed, bidirected, and undirected edges. "
            f"Your graph passed in has the following edge types: {G.edge_types}, whereas "
            f"the function is expecting directed edges named {directed_edge_name}, "
            f"bidirected edges named {bidirected_edge_name}, and undirected edges "
            f"named {undirected_edge_name}."
        )

    if directed_edge_name in G.edge_types:
        if not nx.is_directed_acyclic_graph(G.get_graphs(directed_edge_name)):
            raise nx.NetworkXError("directed edge graph should be directed acyclic")

    union_xyz = x.union(y).union(z)

    # get directed edges
    has_directed = False
    if directed_edge_name in G.edge_types:
        has_directed = True
        G_directed = nx.DiGraph()
        G_directed.add_nodes_from((n, deepcopy(d)) for n, d in G.nodes.items())
        G_directed.add_edges_from(G.get_graphs(edge_type=directed_edge_name).edges)

    # get bidirected edges subgraph
    has_bidirected = False
    if bidirected_edge_name in G.edge_types:
        has_bidirected = True
        G_bidirected = nx.Graph()
        G_bidirected.add_nodes_from((n, deepcopy(d)) for n, d in G.nodes.items())
        G_bidirected.add_edges_from(G.get_graphs(edge_type=bidirected_edge_name).edges)

    # get undirected edges subgraph
    has_undirected = False
    if undirected_edge_name in G.edge_types:
        has_undirected = True
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from((n, deepcopy(d)) for n, d in G.nodes.items())
        G_undirected.add_edges_from(G.get_graphs(edge_type=undirected_edge_name).edges)

    # get ancestral subgraph of x | y | z by removing leaves in directed graph that are not
    # in x | y | z until no more leaves can be removed.
    if has_directed:
        leaves = deque([n for n in G_directed.nodes if G_directed.out_degree[n] == 0])
        while len(leaves) > 0:
            leaf = leaves.popleft()
            if leaf not in union_xyz:
                for p in G_directed.predecessors(leaf):
                    if G_directed.out_degree[p] == 1:
                        leaves.append(p)
                G_directed.remove_node(leaf)

        # remove outgoing directed edges in z
        edges_to_remove = list(G_directed.out_edges(z))
        G_directed.remove_edges_from(edges_to_remove)

    # remove nodes in bidirected graph that are not in x | y | z (since they will be
    # independent due to colliders)
    if has_bidirected:
        nodes = [n for n in G_bidirected.nodes]
        for node in nodes:
            if node not in union_xyz:
                G_bidirected.remove_node(node)

    # remove nodes in undirected graph that are in z to block m-connecting paths
    if has_undirected:
        edges_to_remove = list(G_undirected.edges(z))
        G_undirected.remove_edges_from(edges_to_remove)

    # make new undirected graph from remaining directed, bidirected, and undirected edges
    G_final = nx.Graph()
    if has_directed:
        G_final.add_edges_from(G_directed.edges)
    if has_bidirected:
        G_final.add_edges_from(G_bidirected.edges)
    if has_undirected:
        G_final.add_edges_from(G_undirected.edges)

    disjoint_set = UnionFind(G_final.nodes())
    for component in nx.connected_components(G_final):
        disjoint_set.union(*component)
    disjoint_set.union(*x)
    disjoint_set.union(*y)

    if x and y and disjoint_set[next(iter(x))] == disjoint_set[next(iter(y))]:
        return False
    else:
        return True
