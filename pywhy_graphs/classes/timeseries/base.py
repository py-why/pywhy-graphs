import networkx as nx


class tsdict(dict):
    def __setitem__(self, key, val):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
                f"You passed in {key}."
            )
        if key[1] > 0:
            raise ValueError(f"All lag points should be 0, or less. You passed in {key}.")

        dict.__setitem__(self, key, val)


class BaseTimeSeriesGraph(nx.Graph):
    """A class to imbue undirected graph with time-series structure.

    This should not be used directly.
    """

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict
    adjlist_outer_dict_factory = tsdict
    adjlist_inner_dict_factory = tsdict

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)


class BaseTimeSeriesDiGraph(nx.DiGraph):
    """A class to imbue directed graph with time-series structure.

    This should not be used directly.
    """

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict
    adjlist_outer_dict_factory = tsdict
    adjlist_inner_dict_factory = tsdict

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)


class BaseTimeSeriesMixedEdgeGraph(nx.MixedEdgeGraph):
    """A class to imbue mixed-edge graph with time-series structure.

    This should not be used directly.
    """

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict

    def __init__(self, graphs=None, edge_types=None, **attr):
        if graphs is not None and not all(
            issubclass(graph.__class__, (BaseTimeSeriesGraph, BaseTimeSeriesDiGraph))
            for graph in graphs
        ):
            raise RuntimeError("All graphs for timeseries mixed-edge graph")
        super().__init__(graphs, edge_types, **attr)
