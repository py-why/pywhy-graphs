from collections import defaultdict

import numpy as np

from pywhy_graphs.array import api


def test_array_to_lagged_links():
    max_lag = 3
    n_nodes = 2

    # create a simple time-series array graph
    ts_arr = np.dstack([np.diag(np.ones(n_nodes))] * max_lag)  # .reshape(n_nodes, n_nodes, max_lag)
    ts_arr[..., 0] = 0

    # test that the lagged links are the same as expected
    lagged_links = api.array_to_lagged_links(ts_arr)
    expected_links = defaultdict(list)
    for idx in range(n_nodes):
        for it in range(1, max_lag):
            # tigramite: expected that lags are stored as negative numbers
            expected_links[idx].append(((idx, -it), 1.0))
    assert lagged_links == expected_links
