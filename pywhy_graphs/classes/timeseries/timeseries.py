"""API for networkx-compliant time-series graphs.

"""
import networkx as nx

from .base import BaseTimeSeriesGraph, tsdict


class TimeSeriesGraph(BaseTimeSeriesGraph, nx.Graph):
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


class TimeSeriesDiGraph(BaseTimeSeriesGraph, nx.DiGraph):
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


class TimeSeriesMixedEdgeGraph(BaseTimeSeriesGraph, nx.MixedEdgeGraph):
    """A class to imbue mixed-edge graph with time-series structure.

    This should not be used directly.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = False

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict

    def __init__(self, graphs=None, edge_types=None, max_lag=1, **attr):
        if max_lag is not None:
            if graphs is not None and not all(max_lag == graph.max_lag for graph in graphs):
                raise ValueError(
                    f"Passing in max lag of {max_lag} to time-series mixed-edge graph, but "
                    f"sub-graphs have max-lag of {[graph.max_lag for graph in graphs]}."
                )
        elif graphs is not None:
            # infer max lag
            max_lags = [graph.max_lag for graph in graphs]
            if len(np.unique(max_lags)) != 1:
                raise ValueError(f"All max lags in passed in graphs must be equal: {max_lags}.")
        else:
            max_lag = 1

        if graphs is not None and not all(
            issubclass(graph.__class__, (TimeSeriesGraph, TimeSeriesDiGraph)) for graph in graphs
        ):
            raise RuntimeError("All graphs for timeseries mixed-edge graph")

        attr.update(dict(max_lag=max_lag))
        self.graph = dict()
        self.graph["max_lag"] = max_lag
        super().__init__(graphs, edge_types, **attr)


class StationaryTimeSeriesGraph(TimeSeriesGraph):
    """Stationary time-series graph without directionality on edges.

    This class should not be used directly.

    Included for completeness to enable modeling and working with ``nx.Graph`` like
    objects with time-series structure. By the time-ordering assumption, all lagged
    edges must point forward in time. This serves as an API layer to allow for
    non-directed edges in time (i.e. circular edges among nodes in a ts-PAG).

    Parameters
    ----------
    incoming_graph_data : _type_, optional
        _description_, by default None
    max_lag : int, optional
        _description_, by default 1
    check_time_direction : bool, optional
        _description_, by default True

    See Also
    --------
    StationaryTimeSeriesDiGraph
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = True

    def __init__(
        self, incoming_graph_data=None, max_lag: int = 1, check_time_direction: bool = True, **attr
    ):
        attr.update(dict(max_lag=max_lag, check_time_direction=check_time_direction))
        super(StationaryTimeSeriesGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, **attr
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

    def __init__(
        self, incoming_graph_data=None, max_lag: int = 1, check_time_direction: bool = True, **attr
    ):
        attr.update(dict(max_lag=max_lag, check_time_direction=check_time_direction))
        super(StationaryTimeSeriesDiGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, **attr
        )


class StationaryTimeSeriesMixedEdgeGraph(TimeSeriesMixedEdgeGraph):
    """A mixed-edge causal graph for stationary time-series.

    Parameters
    ----------
    graphs : List of Graph | DiGraph
        A list of networkx single-edge graphs.
    edge_types : List of str
        A list of names for each edge type.
    max_lag : int, optional
        The maximum lag, by default None.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.
    """

    def __init__(self, graphs=None, edge_types=None, max_lag: int = None, **attr):
        attr.update(dict(max_lag=max_lag))
        super().__init__(graphs, edge_types, **attr)

    def copy(self, double_max_lag=True):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        Notes
        -----
        All copies reproduce the graph structure, but data attributes
        may be handled in different ways. There are four types of copies
        of a graph that people might want.

        Deepcopy -- A "deepcopy" copies the graph structure as well as
        all data attributes and any objects they might contain.
        The entire graph object is new so that changes in the copy
        do not affect the original object. (see Python's copy.deepcopy)

        Data Reference (Shallow) -- For a shallow copy the graph structure
        is copied but the edge, node and graph attribute dicts are
        references to those in the original graph. This saves
        time and memory but could cause confusion if you change an attribute
        in one graph and it changes the attribute in the other.
        NetworkX does not provide this level of shallow copy.

        Independent Shallow -- This copy creates new independent attribute
        dicts and then does a shallow copy of the attributes. That is, any
        attributes that are containers are shared between the new graph
        and the original. This is exactly what ``dict.copy()`` provides.
        You can obtain this style copy using:

            >>> G = nx.path_graph(5)
            >>> H = G.copy()
            >>> H = G.copy(as_view=False)
            >>> H = nx.Graph(G)
            >>> H = G.__class__(G)

        Fresh Data -- For fresh data, the graph structure is copied while
        new empty data attribute dicts are created. The resulting graph
        is independent of the original and it has no edge, node or graph
        attributes. Fresh copies are not enabled. Instead use:

            >>> H = G.__class__()
            >>> H.add_nodes_from(G)
            >>> H.add_edges_from(G.edges)

        View -- Inspired by dict-views, graph-views act like read-only
        versions of the original graph, providing a copy of the original
        structure without requiring any memory for copying the information.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data.

        Returns
        -------
        G : Graph
            A copy of the graph.

        See Also
        --------
        to_directed: return a directed copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        G = self.__class__()
        G.graph.update(self.graph)

        if double_max_lag:
            G.graph["max_lag"] = G.max_lag * 2
            for graph in G.get_graphs().values:
                graph.graph["max_lag"] = G.max_lag

        graph_attr = G.graph
        # add all internal graphs to the copy
        for edge_type in self.edge_types:
            graph_func = self._internal_graph_nx_type(edge_type=edge_type)

            if edge_type not in G.edge_types:
                G.add_edge_type(graph_func(**graph_attr), edge_type)

        # add all nodes and edges now
        G.add_nodes_from((n, d.copy()) for n, d in self.nodes.items())
        for edge_type, adj in self.adj.items():
            for u, nbrs in adj.items():
                for v, datadict in nbrs.items():
                    if v[1] == 0:
                        G.add_edge(u, v, edge_type, **datadict.copy())

                    G.add_nodes_from((n, d.copy()) for n, d in self._node.items() if n[1] == 0)
        G.set_auto_removal(None)
        return G
