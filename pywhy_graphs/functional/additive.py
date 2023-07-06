import numpy as np

from pywhy_graphs.typing import Node


def generate_edge_functions_for_node(
    G, node: Node, edge_weight_lims=None, edge_functions=None, random_state=None
):
    r"""Sample edge functions and weights for a given node as a function of their parents.

    This generates edge functions and weights for a given node as a function of their parents,
    which assumes an additive model of the form:

    .. math::
        X = \sum_{j \in parents} w_j f_j(X_j) + \epsilon

    In this function, :math:`w_j` and :math:`f_j` are sampled uniformly at random
    from their respective input lists.

    Parameters
    ----------
    G : Graph
        The causal diagram.
    node : Node
        The node to consider.
    edge_functions : List[Callable[float]], optional
        The set of edge functions that take in an iid sample from the parent and computes
        a transformation (possibly nonlinear), such as ``(lambda x: x**2, lambda x: x)``,
        by default None, which defaults to the identity function ``lambda x: x``.
    edge_weight_lims : Optional[List[float]], optional
        The lower and upper bounds of the edge weight, by default None,
        which defaults to a weight of 1.
    random_state : int, optional
        Random seed, by default None.

    Returns
    -------
    G : Graph
        The graph with edge functions and weights set as node attributes.
    """
    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G
    rng = np.random.default_rng(random_state)

    # get all parents
    parents = directed_G.predecessors(node)

    # sample weight and edge function for each parent
    node_function = dict()
    for parent in parents:
        if parent == node:
            continue

        weight = rng.uniform(low=edge_weight_lims[0], high=edge_weight_lims[1])
        func = rng.choice(edge_functions)
        node_function.update({parent: {"weight": weight, "func": func}})

    # set the node attribute "functions" to hold the weight and function wrt each parent
    G.nodes[node]["parent_functions"] = node_function
    return G
