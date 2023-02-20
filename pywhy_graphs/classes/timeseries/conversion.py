from typing import List

import numpy as np

from pywhy_graphs.typing import Node

from .graph import StationaryTimeSeriesGraph


def tsgraph_to_numpy(G, var_order: List[Node] = None):
    """Convert stationary timeseries graph to numpy array.

    Parameters
    ----------
    G : StationaryTimeSeriesGraph
        A stationary timeseries graph. Can be undirected, or directed.
    var_order : list of Node, optional
        The variable order to order the rows and columns of the first two
        axes of ``ts_graph_arr``.

    Returns
    -------
    ts_graph_arr : ArrayLike of shape (n_variables, n_variables, max_lag + 1)
        The resulting 3D numpy array representing the stationary time-series
        graph. Currently, we do not map different edges to different values.
        The rows are considered the "from nodes" and the columns are considered
        the "to nodes".
    """
    # then we convert this into an array of 1's and 0's
    # we maintain a lagged-order of the nodes, so that way
    # reshaping into a 3D array works properly
    if var_order is None:
        var_order = list(G.variables)
    n_variables = len(var_order)
    max_lag = G.max_lag

    ts_graph_arr = np.zeros((n_variables, n_variables, max_lag + 1))

    for node_idx, node_x in enumerate(var_order):
        for node_jdx, node_y in enumerate(var_order):
            for lag in range(max_lag + 1):
                if G.has_edge((node_x, -lag), (node_y, 0)):
                    ts_graph_arr[node_idx, node_jdx, lag] = 1
    return ts_graph_arr


def numpy_to_tsgraph(arr, var_order: List[Node] = None, create_using=StationaryTimeSeriesGraph):
    """Convert 3D numpy array into a stationary time-series graph.

    Parameters
    ----------
    arr : ArrayLike of shape (n_variables, n_variables, max_lag + 1)
        The resulting 3D numpy array representing the stationary time-series
        graph. The rows are considered the "from nodes" and the columns are considered
        the "to nodes".
    var_order : List[Node], optional
        The variables in order of the rows/columns of the first two axes of ``arr``.
        By default None, which we will then name those nodes ``(0, ..., n_variables)``.
    create_using : PyWhy_Graph graph constructor, optional (default=StationaryTimeSeriesGraph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : StationaryTimeSeriesGraph
        The resulting stationary timeseries graph.
    """
    n_variables, _, max_lag = arr.shape
    if n_variables != arr.shape[1]:
        raise RuntimeError(
            f"The first two axes of ``arr`` should be the number of variables. "
            f"It is {arr.shape} right now."
        )
    max_lag -= 1

    if var_order is None:
        var_order = list(range(n_variables))

    # XXX: do some error checking on the values within arr
    # if not all(val in VALUE_TO_EDGE_MAPPING for val in arr.flatten()):
    #     raise ValueError(f'The 3D stationary timeseries array must only contain values '
    #         f'within our value -> edge mapping: {VALUE_TO_EDGE_MAPPING}.')

    # first we sample the time-series graph
    G = create_using(max_lag=max_lag)
    G.add_variables_from(var_order)

    # now we add edges in according to the array
    for non_lag_node in G.nodes_at(t=0):
        to_idx = var_order.index(non_lag_node[0])
        for lag in range(0, max_lag + 1):
            for lag_node in G.nodes_at(t=lag):
                from_idx = var_order.index(lag_node[0])

                # XXX: improve to allow different edge types
                if arr[from_idx, to_idx, lag] > 0:
                    G.add_edge(lag_node, non_lag_node)

    return G
