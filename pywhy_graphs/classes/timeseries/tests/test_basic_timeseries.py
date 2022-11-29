import pytest

from pywhy_graphs.classes.timeseries.base import BaseTimeSeriesDiGraph, BaseTimeSeriesGraph


@pytest.mark.parametrize("G_func", [BaseTimeSeriesGraph, BaseTimeSeriesDiGraph])
def test_time_nodes(G_func):
    """Basic time-series graphs now store nodes as a tuple."""

    G = G_func()
    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
        (("x1", -3), ("x3", 0)),
    ]
    G.add_edges_from(ts_edges)

    for node in G.nodes:
        assert len(node) == 2
        assert node[1] <= 0

    with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
        G.add_node(1)
    with pytest.raises(ValueError, match="All nodes in time series DAG must be a 2-tuple"):
        G.add_node((1, 2, 3))


@pytest.mark.skip(reason="networkx must implement bug fix.")
def test_func():
    """Testing some stuff... can delete later."""

    from typing import Protocol

    import networkx as nx

    class Mixin(Protocol):
        _adj: Protocol

        def new_method(self):
            pass

    class BaseGraph(nx.DiGraph):
        def __init__(self, incoming_graph_data=None, **attr):
            super().__init__(incoming_graph_data, **attr)

    class Graph(Mixin, BaseGraph):
        def __init__(self, incoming_graph_data=None, **attr):
            super().__init__(incoming_graph_data, **attr)

    ts_edges = [
        (("x1", -1), ("x1", 0)),
        (("x1", -1), ("x2", 0)),
        (("x3", -1), ("x2", 0)),
        (("x3", -1), ("x3", 0)),
    ]
    G = Graph(ts_edges)

    G = nx.to_networkx_graph(ts_edges, create_using=Graph)
    assert G
