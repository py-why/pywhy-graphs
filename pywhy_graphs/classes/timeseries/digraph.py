import networkx as nx

from pywhy_graphs.classes.timeseries.base import BaseTimeSeriesGraph, TsGraphEdgeMixin


class TimeSeriesDiGraph(BaseTimeSeriesGraph, TsGraphEdgeMixin, nx.DiGraph):
    """A class to imbue directed graph with time-series structure.

    See ``BaseTimeSeriesGraph`` for documentation on the
    functionality of time-series graphs.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = False

    # whether to check for valid time-directionality in edges
    check_time_direction: bool = True

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
        self.graph = dict()
        self.graph["max_lag"] = max_lag
        super().__init__(incoming_graph_data=None, **attr)

        # TODO: this is needed because nx.from_edgelist() checks for type of 'create_using',
        # which does not work with Protocol classes
        if incoming_graph_data is not None:
            # we assume a list of tuples of tuples as edges
            if isinstance(incoming_graph_data, list):
                self.add_edges_from(incoming_graph_data)
            elif isinstance(incoming_graph_data, nx.Graph):
                for edge in incoming_graph_data.edges:
                    self.add_edge(*sorted(edge, key=lambda x: x[1]))
            else:
                raise RuntimeError(
                    f"Not implemented yet for incoming graph data that is of "
                    f"type {type(incoming_graph_data)}."
                )


class StationaryTimeSeriesDiGraph(TimeSeriesDiGraph):
    """Stationary time-series directed graph.

    A stationary graph is one where lagged edges repeat themselves
    over time. Edges connecting to nodes in time point "t=0" are
    all the relevant edges needed to depict the time-series graph.

    Time-series graph nodes are defined as a cross-product of variables
    and a time-index. Nodes are always a tuple of variables and the lag.
    For example, a node could be ``('x', -1)`` indicating the 'x' variable
    at '-1' lag.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.
    max_lag : int, optional
        The max lag, by default 1.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    Attributes
    ----------
    stationary : bool
        Whether or not the graph is stationary.
    check_time_direction : bool
        Whether or not to check time directionality is valid, by default True.
        May set to False for undirected graphs.

    Notes
    -----
    A stationary time-series graph is one in which edges over time are repeated.
    In order to properly query for d-separation, one needs to query up to 2 times
    the maximum lag.

    A ts-graph's nodes are defined uniquely by its set of variables and the maximum-lag
    parameter. Given for example, ``('x', 'y', 'z')`` as the set of variables and a
    maximum-lag of 2, then there would be 9 total nodes in the graph consisting of the
    cross-product of ``('x', 'y', 'z')`` and ``(0, 1, 2)``. Nodes are automatically added,
    or deleted depending on the value of the maximum-lag in the graph.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = True

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        super(StationaryTimeSeriesDiGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, max_lag=max_lag, **attr
        )
