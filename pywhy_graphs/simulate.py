import networkx as nx
import numpy as np
import pandas as pd


def simulate_var_process_from_summary_graph(
    G: nx.MixedEdgeGraph, max_lag=1, n_times=1000, random_state: int = None
):
    """Simulate a VAR(max_lag) process starting from a summary graph.

    Parameters
    ----------
    G : nx.MixedEdgeGraph
        A time-series summary graph.
    max_lag : int, optional
        The maximum time-lag to consider, by default 1, which corresponds
        to a VAR(1) process.
    n_times : int
        Number of observations (time points) to simulate, this includes the initial
        observations to start the autoregressive process. By default 1000.
    random_state : int, optional
        The random seed, by default None.

    Returns
    -------
    var_arr : ndarray of shape (n_nodes, n_nodes, max_lag)
        The stationary time-series graph.
    x_df : pandas DataFrame of shape (n_nodes, n_times)
        The sampled dataset.

    Notes
    -----
    Right now, it is assumed that the summary graph is just a DAG.
    """
    rng = np.random.default_rng(random_state)
    n_nodes = G.number_of_nodes()
    var_arr = np.zeros((n_nodes, n_nodes, max_lag))

    # get the non-zeros
    undir_graph = G.to_undirected()

    # simulate weights of the weight matrix
    n_edges = G.number_of_edges()
    summary_arr = np.zeros((n_nodes, n_nodes))
    for edge_type, graph in G.get_graphs().items():
        # get the graph array
        graph_arr = nx.to_numpy_array(graph, weight="weight")
        non_zero_index = np.nonzero(graph_arr)
        weights = rng.normal(size=(len(non_zero_index[0]),))

        # set the weights in the summary graph
        summary_arr[non_zero_index] = weights
    # TODO: generalize to directional weights
    # extract the array and set the weights
    # undir_arr = nx.to_numpy_array(undir_graph, weight="weight")
    # non_zero_index = np.nonzero(undir_arr)
    # weights = rng.normal(size=(n_edges * 2,))
    # print(undir_graph)
    # print(len(weights))
    # print(n_edges)
    # print(undir_arr.shape)
    # print(non_zero_index)
    # undir_arr[non_zero_index] = weights

    # Now simulate across time-points. First initialize such that
    # the edge between every time-point is there and reflective of the
    # summary graph.
    # Assume that every variable has an edge between time points
    for i in range(max_lag):
        var_arr[..., i] = summary_arr

    # simulate initial conditions data
    x0 = rng.standard_normal(size=(n_nodes, max_lag))
    x = np.zeros((n_nodes, n_times))
    x[:, :max_lag] = x0

    # simulate forward in time
    for tdx in range(max_lag, n_times):
        ygen = x[:, tdx]
        for pdx in range(max_lag):
            ygen += np.dot(var_arr[..., pdx], x[:, tdx - pdx - 1].T).T

    # convert to a DataFrame
    x_df = pd.DataFrame(x.T, columns=list(G.nodes))

    return var_arr, x_df
