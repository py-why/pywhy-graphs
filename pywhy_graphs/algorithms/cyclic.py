import networkx as nx


def acyclification(
    G: nx.MixedEdgeGraph,
    directed_edge_type: str = "directed",
    bidirected_edge_type: str = "bidirected",
    copy: bool = True,
) -> nx.MixedEdgeGraph:
    """Acyclify a cyclic graph.

    Applies the acyclification procedure presented in :footcite:`Mooij2020cyclic`.
    This converts to G to what is called :math:`G^{acy}` in the reference.

    Parameters
    ----------
    G : nx.MixedEdgeGraph
        A graph with cycles.
    directed_edge_type : str
        The name of the sub-graph of directed edges.
    bidirected_edge_type : str
        The name of the sub-graph of bidirected edges.
    copy : bool
        Whether to operate on the graph in place, or make a copy.

    Returns
    -------
    G : nx.MixedEdgeGraph
        The acyclified graph.

    Notes
    -----
    This takes
    This replaces all strongly connected components of G by fully connected
    bidirected components without any directed edges. Then any node with an
    edge pointing into the SC (i.e. a directed edge, or bidirected edge) is
    made fully connected with the nodes of the SC either with a directed, or
    bidirected edge.

    References
    ----------
    .. footbibliography::
    """
    if copy:
        G = G.copy()

    # extract the subgraph of directed edges
    directed_G: nx.DiGraph = G.get_graphs(directed_edge_type).copy()
    bidirected_G: nx.Graph = G.get_graphs(bidirected_edge_type).copy()

    # first detect all strongly connected components
    scomps = nx.strongly_connected_components(directed_G)

    # loop over all strongly connected components and their nodes
    for comp in scomps:
        if len(comp) == 1:
            continue

        # extract the parents, or c-components of any node
        # in the strongly-connected component
        scomp_parents = set()
        scomp_c_components = set()
        scomp_children = []

        for node in comp:
            # get any predecessors of SC
            for parent in directed_G.predecessors(node):
                if parent in comp:
                    continue
                scomp_parents.add(parent)

            # get any bidirected edges pointing to elements of SC
            for nbr in bidirected_G.neighbors(node):
                if nbr in comp:
                    continue
                scomp_c_components.add(nbr)

            # keep track of any edges pointing out of the SC
            for child in directed_G.successors(node):
                if child in comp:
                    continue
                scomp_children.append((node, child))

        # first remove all nodes in the cycle
        G.remove_nodes_from(comp)

        # add them back in as a fully connected bidirected graph
        bidirected_fc_G = nx.complete_graph(comp)
        G.add_edges_from(bidirected_fc_G.edges, bidirected_edge_type)

        # add back the children
        G.add_edges_from(scomp_children, directed_edge_type)

        # make all variables connect to the strongly connected component
        for node in comp:
            for parent in scomp_parents:
                G.add_edge(parent, node, directed_edge_type)
            for c_component in scomp_c_components:
                G.add_edge(c_component, node, bidirected_edge_type)
    return G
