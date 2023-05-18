import networkx as nx


def set_node_attributes_with_G(G1, G2, node):
    """Set node attributes in G1 using G2.

    Parameters
    ----------
    G1 : Graph
        The target graph that is modified.
    G2 : Graph
        The source graph that is used to set the node attributes.
    node : Node
        The specific node to set.
    """
    # get the node attributes of node in G2
    src_node_attrs = G2.nodes(data=True)[node]

    # get the target node attrs
    target_node_attrs = G1.nodes(data=True)[node]

    # update the node attributes with respect
    target_node_attrs.update(src_node_attrs)

    nx.set_node_attributes(G1, {node: target_node_attrs})
    return G1
