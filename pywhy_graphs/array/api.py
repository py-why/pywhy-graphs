def remove_ts_edge(arr, arr_idx, u_node, v_node, t=None, homologous=True):
    
    pass



def get_summary_graph(arr):
    """Compute the time-series summary graph from the given time-series graph.

    The summary graph is defined as a graph where nodes are the variables in the
    multivariate time-series, and there is an edge betwen two nodes if there is any
    edge between the two nodes in the full time-series graph.

    Parameters
    ----------
    arr : ndarray of shape (n_nodes, n_nodes, max_lag)
        The full time-series graph, where endpoints are encoded via some
        enumeration. By default, the enumeration is causal-learn.

    Returns
    -------
    summary_arr : ndarray of shape (n_nodes, n_nodes)
        The summary graph.
    """
    pass

