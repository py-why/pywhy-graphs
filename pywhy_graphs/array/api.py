from typing import Dict, List, Optional, Set

import numpy as np
from numpy.typing import NDArray

from pywhy_graphs.config import ARRAY_ENUMS
from pywhy_graphs.typing import Node


def _check_valid_ts_arr(arr, arr_enum=None):
    if arr.ndim != 3:
        raise RuntimeError(
            f"There should be 3 dimensions in time-series graph, but your array "
            f"has {arr.ndims} dimensions."
        )
    if arr.shape[0] != arr.shape[1]:
        raise RuntimeError(
            f"This array is not a valid time-series graph because the first two dimensions "
            f"of {arr.shape} do not match."
        )

    if arr_enum is not None:
        arr_vals = np.unique(arr)
        if any(val not in arr_enum for val in arr_vals):
            raise RuntimeError(f"Array values should follow enumeration of {arr_enum}.")

    # all time-series arrays are assumed to have rows as the starting node
    # and the columns as the ending node


def check_edge_homologous(arr, u_node, v_node, t):
    pass


def get_summary_graph(arr: NDArray, arr_enum: str = "clearn"):
    """Compute the time-series summary graph from the given time-series graph.

    The summary graph is defined as a graph where nodes are the variables in the
    multivariate time-series, and there is an edge between two nodes if there is any
    edge between the two nodes in the full time-series graph.

    Parameters
    ----------
    arr : ndarray of shape (n_nodes, n_nodes, max_lag)
        The full time-series graph, where endpoints are encoded via some
        enumeration. By default, the enumeration is causal-learn.
    arr_enum : str
        The enumeration for values to use for edges. By default, 'clearn',
        for the ``causal-learn`` package enumerated values.

    Returns
    -------
    summary_arr : ndarray of shape (n_nodes, n_nodes)
        The summary graph.
    """
    arr_enum = ARRAY_ENUMS[arr_enum]  # type: ignore
    _check_valid_ts_arr(arr, arr_enum)

    n_nodes, _, _ = arr.shape
    summary_arr = np.zeros((n_nodes, n_nodes))

    for u_node in range(n_nodes):
        for v_node in range(n_nodes):
            if u_node == v_node:
                continue

            # get the edges that are present for (u, v)
            # Ex: u -> v and u <-> v
            arr_vals = np.unique(arr[u_node, v_node, :])

            # get the enumeration value
            # TODO: fix for enumerations
            for val in arr_vals:
                if val == 0:
                    continue

            # set summary array at (u, v) to the enumeration value
            summary_arr[u_node, v_node] = arr_enum
    return summary_arr


def array_to_lagged_links(
    arr: NDArray, arr_idx: Optional[List[Node]] = None, include_weights: bool = True
) -> Dict[Node, List[Set]]:
    """Convert a time-series 3D array to a dictionary of lagged links.

    For usage with tigramite. Note that by assumption, if we have a stationary
    causal graph, then we simply need to model all the connections relative to
    a single time point (t=0). Connections at t=0 correspond to contemporaneous
    edges, while connections at t<0 correspond to lagged links.

    Parameters
    ----------
    arr : NDArray of shape (n_nodes, n_nodes, max_lag + 1)
        The time-series graph array with elements as linear coefficients.
        This is a VAR process connectivity matrix. The first element of
        the time-lag axis corresponds to connectivity at the same time point,
        or known as contemporaneous edges in time-series graphs.
    arr_idx : list of length n_nodes, optional
        A list of length 'n_nodes', specifying the names of each node, in
        the order of which ``arr`` is laid out. If None (default), then
        the output node names will be integer values starting from '0'
        ending with 'n_nodes'.

    Returns
    -------
    lagged_links : Dict[Node, List[Set[Node, int]]]
        Dictionary of form linking nodes with other nodes at time-lags
        and coefficients. The time-lags are integer values 0 or less.
        See Notes for details.

    Notes
    -----
    Lagged links in Tigramite v5.1.0.1 are encoded as follows:
    ``{'node0': [(('node0', -1), coeff), ...], 'node1': [...], ...}``, where
    it is a dictionary with nodes as keys, and a list as values. Within
    the list are sets of node connections, time lags and coefficient values
    for the linear weight from the time-lagged node to the keyed node.
    """
    _check_valid_ts_arr(arr)
    n_nodes, _, max_lag = arr.shape
    max_lag -= 1

    if arr_idx is None:
        arr_idx = np.arange(n_nodes)

    lagged_links: Dict[Node, List] = dict()

    # loop over all nodes and extract its adjacencies
    for idx, node in enumerate(range(n_nodes)):
        node_name = arr_idx[idx]
        lagged_links[node_name] = []

        # for every lag point, find neighbors
        for it in range(max_lag + 1):
            node_nbrs = np.argwhere(arr[node, :, it] != 0).flatten()

            # append sets of nbrs, lag point and coefficient
            for nbr in node_nbrs:
                coeff = arr[node, nbr, it].item()
                if include_weights:
                    new_item = ((nbr, -it), coeff)
                else:
                    new_item = (nbr, -it)
                lagged_links[node_name].append(new_item)
    return lagged_links


def remove_ts_edge(arr, arr_idx, u_node, v_node, t=None):
    if t is None:
        t = np.arange(arr.shape[-1], dtype=int)

    u_idx = np.argwhere(arr_idx == u_node)
    v_idx = np.argwhere(arr_idx == v_node)
    arr[u_idx, v_idx, t] = 0


if __name__ == "__main__":
    graph = np.array([[[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.0, 0.1, 0.0], [0.3, 0.0, 0.0]]])

    print(graph.shape)
    print(graph)
    print(graph[..., 0])
    print(graph[1, 0, ...])
