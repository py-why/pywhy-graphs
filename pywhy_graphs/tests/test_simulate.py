import pandas as pd
import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.simulate import (
    simulate_linear_var_process,
    simulate_random_er_dag,
    simulate_var_process_from_summary_graph,
)


@pytest.mark.parametrize(
    ["n_variables", "max_lag", "n_times", "n_realizations"],
    [
        [1, 1, 1, 1],
        [5, 2, 10, 3],
    ],
)
def test_simulate_var(n_variables, max_lag, n_times, n_realizations):
    data, graph = simulate_linear_var_process(
        n_variables=n_variables, max_lag=max_lag, n_times=n_times, n_realizations=n_realizations
    )

    assert len(graph.variables) == n_variables
    assert graph.max_lag == max_lag

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (n_times * n_realizations, n_variables)


@pytest.mark.parametrize(
    ["n_variables", "max_lag", "n_times"],
    [
        [1, 1, 1],
        [5, 2, 10],
    ],
)
def test_simulate_summary_graph(n_variables, max_lag, n_times):
    random_state = 12345

    # simulate random graph
    G = simulate_random_er_dag(n_nodes=n_variables, seed=random_state, ensure_acyclic=False)

    # simulate now from the resulting graph
    G = pywhy_nx.MixedEdgeGraph([G], edge_types=["directed"])
    data, graph = simulate_var_process_from_summary_graph(
        G, max_lag=max_lag, n_times=n_times, random_state=random_state
    )

    assert G.number_of_nodes() == n_variables
    assert len(graph.variables) == n_variables
    assert graph.max_lag == max_lag

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (n_times, n_variables)
