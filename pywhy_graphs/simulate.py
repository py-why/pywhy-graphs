import networkx as nx 
import numpy as np
import pywhy_graphs


def simulate_var_process_from_summary_graph(G: nx.MixedEdgeGraph, max_lag=1, random_state: int=None):
    rng = np.random.default_rng(random_state)
    n_nodes = G.number_of_nodes()
    var_arr = np.zeros((n_nodes, n_nodes, max_lag))

    # get the non-zeros
    undir_graph = G.to_undirected()

    # simulate weights of the weight matrix
    n_edges = G.number_of_edges()
    weights = rng.normal(size=(n_edges, 1))

    # extract the array and set the weights
    undir_arr = nx.to_numpy_array(undir_graph, weight='weight')
    undir_arr[undir_arr != 0] = weights

    # Now simulate across time-points. First initialize such that
    # the edge between every time-point is there and reflective of the
    # summary graph.
    # Assume that every variable has an edge between time points
    for i in range(max_lag):
        var_arr[..., i] = undir_arr

    return var_arr
    


def main():
    # define a summary graph
    directed_edges = nx.DiGraph(
        [
            ("x8", "x2"),
            ("x9", "x2"),
            ("x10", "x1"),
            ("x2", "x4"),
            ("x4", "x6"),  # start of cycle
            ("x6", "x5"),
            ("x5", "x3"),
            ("x3", "x4"),  # end of cycle
            ("x6", "x7"),
        ]
    )
    bidirected_edges = nx.Graph([("x1", "x3")])
    G = nx.MixedEdgeGraph([directed_edges, bidirected_edges], ["directed", "bidirected"])

    # generate data

    acyclic_G = pywhy_graphs.acyclification(G)


if __name__ == '__main__':
    main()