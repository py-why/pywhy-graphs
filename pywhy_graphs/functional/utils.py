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


def _preprocess_parameter_inputs(node_mean_lims, node_std_lims, edge_functions, edge_weight_lims):
    """Helper function to preprocess common parameter inputs for sampling functional graphs.

    Nodes' exogenous variables are sampled from a Gaussian distribution.
    Edges are sampled, such that an additive linear model is assumed. Note
    the edge functions may be nonlinear, but how they are combined for each
    node as a function of its parents is linear.
    """
    if node_mean_lims is None:
        node_mean_lims = [0, 0]
    elif len(node_mean_lims) != 2:
        raise ValueError("node_mean_lims must be a list of length 2.")
    if node_std_lims is None:
        node_std_lims = [1, 1]
    elif len(node_std_lims) != 2:
        raise ValueError("node_std_lims must be a list of length 2.")
    if edge_functions is None:
        edge_functions = [lambda x: x]
    if edge_weight_lims is None:
        edge_weight_lims = [1, 1]
    elif len(edge_weight_lims) != 2:
        raise ValueError("edge_weight_lims must be a list of length 2.")

    return node_mean_lims, node_std_lims, edge_functions, edge_weight_lims
