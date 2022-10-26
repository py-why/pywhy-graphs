"""API for networkx-compliant time-series graphs.

"""
from typing import List, Set

import networkx as nx
import numpy as np
from networkx import NetworkXError

from pywhy_graphs.typing import Node


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


class TimeSeriesGraphMixin:
    """A mixin class for time-series graph.

    Adds the distinction between variables and nodes in a networkx-compliant
    graph. In addition, adds a ``max_lag`` parameter to keep track of how
    far back in terms of the time-index to keep track of.

    Also, adds distinction between contemporaneous and lagged edges, as well
    as contemporaneous and lagged neighbors.
    """

    @property
    def max_lag(self) -> int:
        """The maximum time-index lag."""
        return self.graph["max_lag"]

    @property
    def variables(self) -> Set[Node]:
        """Set of variables in the time-series.

        Nodes in a time-series graph consist of variables X times.

        Returns
        -------
        variables : Set[Node]
            A set of variables.
        """
        node_vars = set()
        for node in self.nodes:
            node_vars.add(node[0])
        return node_vars

    @property
    def contemporaneous_edges(self) -> List:
        """List of instantaneous (i.e. at same time point) edges."""
        edges = []
        for u_edge, v_edge in self.edges:
            if u_edge[1] == 0 and v_edge[1] == 0:
                edges.append((u_edge, v_edge))
        return edges

    @property
    def lag_edges(self) -> List:
        """List of lagged edges."""
        edges = []
        for u_edge, v_edge in self.edges:
            if u_edge[1] < 0 and v_edge[1] == 0:
                edges.append((u_edge, v_edge))
        return edges

    @property
    def non_lag_nodes(self) -> Set:
        """Nodes at t=0."""
        nodes = set()
        for node in self.nodes:
            if node[1] == 0:
                nodes.add(node)
        return nodes

    def lagged_neighbors(self, u):
        """Neighbors from t < u's current time index."""
        nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] < 0]

    def contemporaneous_neighbors(self, u):
        """Neighbors from the same time index as u."""
        nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] == 0]


class StationaryTimeSeriesMixin:
    _auto_removal: bool = True

    def _check_ts_node(self, node):
        if not isinstance(node, tuple) or len(node) != 2:
            raise ValueError(
                f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
                f"You passed in {node}."
            )
        if node[1] > 0:
            raise ValueError(f"All lag points should be 0, or less. You passed in {node}.")
        if node[1] < -self.max_lag:
            raise ValueError(f"Lag {node[1]} cannot be greater than set max_lag {self.max_lag}.")

    def add_variable(self, var_name):
        pass

    def add_variables_from(self, var_names):
        pass

    def add_node(self, node_name, **attr):
        self._check_ts_node(node_name)
        super().add_node(node_name, **attr)
        var_name, _ = node_name

        for t in range(self.max_lag + 1):
            super().add_node((var_name, -t), **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for node in nodes_for_adding:
            newdict = attr.copy()

            if isinstance(node[0], tuple):
                ndict = node[1]
                node = node[0]
                newdict = attr.copy()
                newdict.update(ndict)
            self.add_node(node, **newdict)

    def remove_node(self, node_name):
        self._check_ts_node(node_name)
        var_name, _ = node_name
        super().remove_node(node_name)
        if self._auto_removal:
            for t in range(self.max_lag + 1):
                try:
                    super().remove_node((var_name, -t))
                except NetworkXError:
                    continue

    def remove_nodes_from(self, ebunch):
        for node in ebunch:
            self.remove_node(node)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if v_lag != 0:
            raise ValueError(f'The lag of the "to node" {v_lag} should be 0.')
        if v_lag < u_lag:
            raise RuntimeError(
                f'The lag of the "to node" {v_lag} should be greater than "from node" {u_lag}'
            )
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)

        if u_lag < 0:
            # add all homologous edges
            # v_lag_dist = 0 - v_lag
            # u_lag = v_lag_dist - u_lag
            self._add_homologous_ts_edges(u, v, u_lag, **attr)
        elif u_lag == 0:
            # add all homologous contemporaneous edges
            self._add_homologous_contemporaneous_edges(u, v, **attr)

    def add_edges_from(self, ebunch, **attr):
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
            dd.update(attr)
            self.add_edge(u, v, **dd)

    def remove_edge(self, u_of_edge, v_of_edge):
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if v_lag != 0:
            raise RuntimeError('The lag of the "from" node should be 0.')

        if self._auto_removal:
            if u_lag < 0:
                # remove all homologous edges
                self._remove_homologous_ts_edges(u, v, u_lag)
            elif u_lag == 0:
                # remove all contemporaneous homologous edges
                self._remove_homologous_contemporaneous_edges(u, v)
        else:
            super().remove_edge(u_of_edge, v_of_edge)

    def remove_edges_from(self, ebunch):
        for edge in ebunch:
            self.remove_edge(*edge)

    def _add_homologous_contemporaneous_edges(self, u, v, **attr):
        for t in range(self.max_lag + 1):
            super().add_edge((u, -t), (v, -t), **attr)

    def _add_homologous_ts_edges(self, u, v, lag, **attr):
        lag = np.abs(lag)

        to_t = 0
        for from_t in range(lag, self.max_lag + 1, lag):
            super().add_edge((u, -from_t), (v, -to_t), **attr)
            to_t = from_t

    def _remove_homologous_contemporaneous_edges(self, u, v):
        for t in range(self.max_lag + 1):
            super().remove_edge((u, -t), (v, -t))

    def _remove_homologous_ts_edges(self, u, v, lag):
        lag = np.abs(lag)

        to_t = 0
        for from_t in range(lag, self.max_lag + 1, lag):
            super().remove_edge((u, -from_t), (v, -to_t))
            to_t = from_t

    def set_auto_removal(self, auto):
        self._auto_removal = auto

    def copy(self, as_view: bool = False):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        If `as_view` is True then a view is returned instead of a copy.

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
        and the original. This is exactly what `dict.copy()` provides.
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
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items() if n[1] == 0)
        G.add_edges_from(
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
            if v[1] == 0
        )
        G.set_auto_removal(False)
        return G


class TimeSeriesGraph(nx.Graph, TimeSeriesGraphMixin):
    """A class to imbue undirected graph with time-series structure."""

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict
    adjlist_outer_dict_factory = tsdict
    adjlist_inner_dict_factory = tsdict

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
        super().__init__(incoming_graph_data, **attr)


class TimeSeriesDiGraph(nx.DiGraph, TimeSeriesGraphMixin):
    """A class to imbue directed graph with time-series structure."""

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict
    adjlist_outer_dict_factory = tsdict
    adjlist_inner_dict_factory = tsdict

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
        super().__init__(incoming_graph_data, **attr)


class StationaryTimeSeriesGraph(StationaryTimeSeriesMixin, TimeSeriesGraph):
    """Stationary time-series undirected graph.

    Included for completeness to enable modeling and working with ``nx.Graph`` like
    objects with time-series structure. By the time-ordering assumption, all lagged
    edges must point forward in time.
    """

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        super().__init__(incoming_graph_data, max_lag=max_lag, **attr)


class StationaryTimeSeriesDiGraph(StationaryTimeSeriesMixin, TimeSeriesDiGraph):
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
    """

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        super().__init__(incoming_graph_data, max_lag=max_lag, **attr)


class StationaryTimeSeriesMixedEdgeGraph(nx.MixedEdgeGraph):
    def __init__(self, graphs=None, edge_types=None, **attr):
        super().__init__(graphs, edge_types, **attr)

    def add_edge(self, u_of_edge, v_of_edge, edge_type, **attr):
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if v_lag != 0:
            raise RuntimeError('The lag of the "from" node should be 0.')

        if u_lag < 0:
            # add all homologous edges
            self._add_homologous_ts_edges(u, v, u_lag, edge_type, **attr)
        elif u_lag == 0:
            # add all homologous contemporaneous edges
            self._add_homologous_contemporaneous_edges(u, v, edge_type, **attr)

    def add_edges_from(self, ebunch_to_add, edge_type, **attr):
        for ebunch in ebunch_to_add:
            self.add_edge(ebunch[0], ebunch[1], edge_type, **attr)

    def remove_edge(self, u_of_edge, v_of_edge, edge_type):
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if v_lag != 0:
            raise RuntimeError('The lag of the "from" node should be 0.')

        if u_lag < 0:
            # remove all homologous edges
            self._remove_homologous_ts_edges(u, v, u_lag, edge_type)
        elif u_lag == 0:
            # remove all contemporaneous homologous edges
            self._remove_homologous_contemporaneous_edges(u, v, edge_type)

    def remove_edges_from(self, ebunch, edge_type):
        for edge in ebunch:
            self.remove_edge(*edge, edge_type)

    def _add_homologous_contemporaneous_edges(self, u, v, edge_type, **attr):
        for t in range(self.max_lag + 1):
            super().add_edge((u, t), (v, t), edge_type, **attr)

    def _add_homologous_ts_edges(self, u, v, lag, edge_type, **attr):
        if lag <= 0:
            raise RuntimeError(f"If lag is {lag}, then add contemporaneous edges.")

        to_t = 0
        for from_t in range(lag, self.max_lag + 1, lag):
            super().add_edge((u, from_t), (v, to_t), edge_type, **attr)
            to_t = from_t

    def _remove_homologous_contemporaneous_edges(self, u, v, edge_type):
        for t in range(self.max_lag + 1):
            super().remove_edge((u, t), (v, t), edge_type)

    def _remove_homologous_ts_edges(self, u, v, lag, edge_type):
        if lag <= 0:
            raise RuntimeError(f"If lag is {lag}, then remove contemporaneous edges.")

        to_t = 0
        for from_t in range(lag, self.max_lag + 1, lag):
            super().remove_edge((u, from_t), (v, to_t), edge_type)
            to_t = from_t


def complete_ts_graph(
    variables,
    max_lag: int,
    include_contemporaneous: bool = True,
    create_using=StationaryTimeSeriesGraph,
) -> StationaryTimeSeriesDiGraph:
    G = create_using(max_lag=max_lag)

    # add all possible edges
    for node in variables:
        for to_node in variables:
            for lag in range(max_lag + 1):
                # skip contemporaneous edges if necessary
                if not include_contemporaneous and lag == 0:
                    continue
                # do not add self connections
                if node == to_node and lag == 0:
                    continue
                # do not add cyclicity
                if lag == 0 and (
                    G.has_edge((node, 0), (to_node, 0)) or G.has_edge((to_node, 0), (node, 0))
                ):
                    continue
                # if there is already an edge, do not add
                if G.has_edge((node, -lag), (to_node, 0)):
                    continue
                G.add_edge((node, -lag), (to_node, 0))
    return G


def empty_ts_graph(
    variables, max_lag, create_using=StationaryTimeSeriesGraph
) -> StationaryTimeSeriesDiGraph:
    G = create_using(max_lag=max_lag)
    for node in variables:
        G.add_node((node, 0))
    return G
