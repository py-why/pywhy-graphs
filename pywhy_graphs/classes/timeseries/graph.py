import networkx as nx

from pywhy_graphs.classes.timeseries.base import BaseTimeSeriesGraph, TsGraphEdgeMixin


class TimeSeriesGraph(BaseTimeSeriesGraph, TsGraphEdgeMixin, nx.Graph):
    """A class to imbue undirected graph with time-series structure.

    This should not be used directly. See ``BaseTimeSeriesGraph`` for documentation on the
    functionality of time-series graphs.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = False

    # whether to check for valid time-directionality in edges
    check_time_direction: bool = False

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
        self.graph = dict()
        self.graph["max_lag"] = max_lag
        super(TimeSeriesGraph, self).__init__(incoming_graph_data=None, **attr)

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


class StationaryTimeSeriesGraph(TimeSeriesGraph):
    """Stationary time-series graph without directionality on edges.

    This class should not be used directly.

    Included for completeness to enable modeling and working with ``nx.Graph`` like
    objects with time-series structure. By the time-ordering assumption, all lagged
    edges must point forward in time. This serves as an API layer to allow for
    non-directed edges in time (i.e. circular edges among nodes in a ts-PAG).

    Parameters
    ----------
    incoming_graph_data : iterable, optional
        The graph data to set, by default None.
    max_lag : int, optional
        Maximum lag, by default 1.

    See Also
    --------
    StationaryTimeSeriesDiGraph
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = True

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        super(StationaryTimeSeriesGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, max_lag=max_lag, **attr
        )
