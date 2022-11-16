from typing import Callable, List

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from pywhy_graphs.classes import StationaryTimeSeriesDiGraph
from pywhy_graphs.typing import TsNode


def simulate_data_from_var(
    var_arr: NDArray,
    n_times: int = 1000,
    n_realizations: int = 1,
    var_names: List[TsNode] = None,
    random_state: int = None,
):
    """Simulate data from an already set VAR process.

    Parameters
    ----------
    var_arr : ndarray of shape (n_variables, n_variables, max_lag)
        The stationary time-series vector-auto-regressive process.
    n_times : int, optional
        Number of observations (time points) to simulate, this includes the initial
        observations to start the autoregressive process. By default 1000.
    var_names : list of nodes, optional
        The variable names. If passed in, then must have length equal ``n_variables``. If passed in,
        then the output will be converted to a pandas DataFrame with ``var_names`` as the
        columns. By default, None.
    random_state : int, optional
        The random state, by default None.

    Returns
    -------
    x : NDArray of shape (n_variables, n_times), or pandas.DataFrame of shape (n_times, n_variables)
        The simulated data. If ``node_names`` are passed in, then the output will be
        converted to a pandas DataFrame.

    Notes
    -----
    The simulated ``x`` array consists of multiple "instances" of the underlying stationary
    VAR process. For example, if ``n_times`` is 1000, and ``max_lag = 2``, then technically you
    have 500 realizations of the time-series graph occurring over this multivariate time-series.
    However, each realization is dependent on the previous realizations in this case.

    In order to start from a completely independent initial conditions, then one can modify the
    ``n_realizations`` parameter instead. To generate 500 independent realizations in the
    above example, one would set ``n_realizations = 500`` and ``n_times=2``.
    """
    if var_arr.shape[0] != var_arr.shape[1]:
        raise ValueError(
            f"The shape of the VAR array should be "
            f"(n_variables, n_variables, max_lag). Your array dimensions are {var_arr.shape}."
        )
    n_vars, _, max_lag = var_arr.shape

    if var_names is not None and len(var_names) != n_vars:
        raise ValueError(f"The passed in node names {var_names} should have {n_vars} variables.")

    rng = np.random.default_rng(random_state)

    # initialize the final data array for speed
    x = np.zeros((n_vars, n_realizations * n_times))

    for idx in range(n_realizations):
        # sample the initial conditions
        x0 = rng.standard_normal(size=(n_vars, max_lag))
        x[:, :max_lag] = x0

        # simulate forward in time
        for tdx in range(max_lag, n_times):
            # note that for a single realization, this is just 'tdx'
            starting_point = (idx + 1) * tdx
            ygen = x[:, starting_point]

            # loop over the lags to generate the next sample as a linear
            # combination of previous lag data points
            for pdx in range(max_lag):
                ygen += np.dot(var_arr[..., pdx], x[:, tdx - pdx - 1].T).T

    # convert to a DataFrame, if node names are explicitly passed in
    if var_names is not None:
        x = pd.DataFrame(x.T, columns=var_names)
    return x


def simulate_linear_var_process(
    n_variables: int = 5,
    p_time_self: float = 0.5,
    p_time_vars: float = 0.5,
    p_contemporaneous: float=0.5,
    max_lag: int = 1,
    n_times: int = 1000,
    n_realizations: int = 1,
    weight_dist: Callable = scipy.stats.norm,
    random_state: int = None,
) -> NDArray:
    """Simulate a linear VAR process of a "stationary" causal graph.

    Parameters
    ----------
    n_variables : int, optional
        The number of variables to included in the final simulation, by default 5.
    p_time_self : float, optional
        The probability for an edge across time for the same variable, by default 0.5.
    p_time_vars : float, optional
        The probability for an edge across time for different variables, by default 0.5.
    p_contemporaneous : float, optional
        The probability for a contemporaneous edge among different variables, by default 0.5.
        Can be set to 0 to specify that the underlying causal process has no instantaneous
        effects.
    max_lag : int, optional
        The maximum lag allowed in the resulting process, by default 1.
    n_times : int, optional
        The number of time points to generate per realization, by default 1000. See
        `simulate_data_from_var` for more information.
    n_realizations : int, optional
        The number of independent realizations, by default 1. See `simulate_data_from_var` for
        more information.
    weight_dist : Callable, optional
        The distribution of weights for connections, by default None.
    random_state : int, optional
        The random state, by default None.

    Returns
    -------
    data : NDArray of shape (n_nodes, n_times * n_realizations)
        The simulated data.
    var_model : StationaryTimeSeriesDiGraph of shape (n_nodes, n_nodes, max_lag)
        The resulting time-series causal graph.

    Notes
    -----
    In stationary time-series graphs without latent confounders, there is never a worry
    about acyclicity among "lagged nodes" in the graph. However, if we model the situation
    where there are instantaenous, or contemporaneous effects, then those edges may form
    a cycle when simulating the graph. Therefore, if ``p_contemporaneous > 0``, then an
    additional check for DAG among the contemporaneous edge network is checked.

    To simulate VAR process with latent confounders (i.e. missing variable time-series), then
    one can simply simulate the full VAR process and then delete the simulated time-series data
    from the latent variable.
    """
    rng = np.random.default_rng(random_state)

    # first we sample the time-series graph
    node_names = range(n_variables)
    G = StationaryTimeSeriesDiGraph(max_lag=max_lag)
    G.add_nodes_from(node_names)

    # loop through all possible edge combinations from (x, 0) to (x', -lag)
    # for lag up to max_lag
    for non_lag_node in G.nodes_at(t=0):
        for lag in range(1, max_lag + 1):
            for lag_node in G.nodes_at(t=-lag):
                # then we are looking at a auto-lag nbr in the same variable
                if non_lag_node[1] == lag_node[1] and p_contemporaneous > 0:
                    if rng.binomial(n=1, p=p_contemporaneous, size=None) == 1:
                        G.add_edge(lag_node, non_lag_node)
                    
                    # check that the addition of this edge does not result in a cyclic
                    # causal relationship
                    if not nx.is_directed_acyclic_graph(G.subgraph(G.nodes_at(t=0))):
                        G.remove_edge(lag_node, non_lag_node)
                elif non_lag_node[0] == lag_node[0] and p_time_self > 0:
                    # sample binomial distribution with probability p_time_self
                    if rng.binomial(n=1, p=p_time_self, size=None) == 1:
                        G.add_edge(lag_node, non_lag_node)
                elif p_time_vars > 0:
                    if rng.binomial(n=1, p=p_time_vars, size=None) == 1:
                        G.add_edge(lag_node, non_lag_node)

    # then we convert this into an array of 1's and 0's
    # we maintain a lagged-order of the nodes, so that way
    # reshaping into a 3D array works properly
    var_order = list(G.variables)
    nodelist = []
    for variable in var_order:
        for lag in range(G.max_lag + 1):
            nodelist.append((variable, -lag))
    graph_arr = nx.to_numpy_array(G, weight="weight", nodelist=nodelist)

    # reshape the array to the correct shape
    graph_arr = graph_arr.reshape((n_variables, n_variables, max_lag))
    nnz_index = np.nonzero(graph_arr)
    nnz = len(nnz_index)

    # we sample weights from our weight distribution to fill
    # in every non-zero index of our VAR array
    weights = weight_dist(nnz)
    graph_arr[nnz_index] = weights

    # our resulting VAR array is the function, which we will
    # simulate our data, starting from random Gaussian initial conditions.
    x = simulate_data_from_var(
        var_arr=graph_arr,
        n_times=n_times,
        n_realizations=n_realizations,
        var_names=nodelist,
        random_state=random_state,
    )
    return x, G


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
    # undir_graph = G.to_undirected()

    # simulate weights of the weight matrix
    # n_edges = G.number_of_edges()
    summary_arr = np.zeros((n_nodes, n_nodes))
    for _, graph in G.get_graphs().items():
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
