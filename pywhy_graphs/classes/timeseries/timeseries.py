"""API for networkx-compliant time-series graphs.

"""
from copy import copy
from typing import Dict, Iterator, List, Optional

import numpy as np
from networkx import NetworkXError
from networkx.classes.graph import _CachedPropertyResetterAdj

from .base import BaseTimeSeriesDiGraph, BaseTimeSeriesGraph, BaseTimeSeriesMixedEdgeGraph


class TsGraphEdgeMixin:
    """A mixin class for time-series graph edges.

    Adds the distinction between variables and nodes in a networkx-compliant
    graph. In addition, adds a ``max_lag`` parameter to keep track of how
    far back in terms of the time-index to keep track of.

    Also, adds distinction between contemporaneous and lagged edges, as well
    as contemporaneous and lagged neighbors.
    """

    _auto_removal: Optional[str]
    graph: Dict
    _adj: _CachedPropertyResetterAdj
    edges: Iterator

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if u_lag > 0 or v_lag > 0:
            raise RuntimeError(f"All lags should be negative or 0, not {u_lag} or {v_lag}.")
        if self.check_time_direction:
            if v_lag < u_lag:
                raise RuntimeError(
                    f'The lag of the "to node" {v_lag} should be greater than "from node" {u_lag}'
                )
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)

        if self._auto_addition is True:
            if u_lag < 0:
                # add all homologous edges
                # v_lag_dist = 0 - v_lag
                # u_lag = v_lag_dist - u_lag
                self._add_homologous_ts_edges(u, v, u_lag, v_lag, **attr)
            elif u_lag == 0:
                # add all homologous contemporaneous edges
                self._add_homologous_contemporaneous_edges(u, v, **attr)
        else:
            super().add_edge(u_of_edge, v_of_edge, **attr)

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

    def remove_edge(self, u_of_edge, v_of_edge, check_lag: bool = False):
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if v_lag != 0 and check_lag:
            raise RuntimeError(f'The lag of the "to" node, {v_of_edge} should be 0.')

        if self._auto_removal is not None:
            if u_lag == v_lag:
                # remove all contemporaneous homologous edges
                if self._auto_removal == "backwards":
                    self._backward_remove_homologous_contemporaneous_edges(u, v)
                elif self._auto_removal == "forwards":
                    self._forward_remove_homologous_contemporaneous_edges(u, v, u_lag)
            elif u_lag < 0:
                # remove all homologous edges
                if self._auto_removal == "backwards":
                    self._backward_remove_homologous_ts_edges(u, v, u_lag, v_lag)
                elif self._auto_removal == "forwards":
                    self._forward_remove_homologous_ts_edges(u, v, u_lag, v_lag)
        else:
            super().remove_edge(u_of_edge, v_of_edge)  # type: ignore

    def remove_edges_from(self, ebunch):
        for edge in ebunch:
            self.remove_edge(*edge)

    def _add_homologous_contemporaneous_edges(self, u, v, **attr):
        """Add homologous edges to all contemporaneous pairs."""
        for t in range(self._max_lag + 1):
            super().add_edge((u, -t), (v, -t), **attr)

    def _add_homologous_ts_edges(self, u, v, u_lag, v_lag, **attr):
        """Add homologous time-series edges in a backwards manner."""
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        to_t = v_lag
        from_t = u_lag
        for _ in range(u_lag, self._max_lag + 1):
            super().add_edge((u, -from_t), (v, -to_t), **attr)
            to_t += 1
            from_t += 1

    def _backward_remove_homologous_contemporaneous_edges(self, u, v):
        """Remove homologous time-series edges in the backwards direction."""
        for t in range(self._max_lag + 1):
            super().remove_edge((u, -t), (v, -t))

    def _forward_remove_homologous_contemporaneous_edges(self, u, v, lag):
        """Remove homologous time-series edges in the forward direction."""
        for t in range(lag, 0, -1):
            super().remove_edge((u, -t), (v, -t))

    def _backward_remove_homologous_ts_edges(self, u, v, u_lag, v_lag):
        """Remove homologous time-series edges in the backwards direction."""
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        if u_lag <= v_lag:
            raise RuntimeError(f"From lag {u_lag} should be larger than to lag {v_lag}.")

        from_t = u_lag
        to_t = v_lag
        for _ in range(u_lag, self._max_lag + 1):
            super().remove_edge((u, -from_t), (v, -to_t))
            from_t += 1
            to_t += 1

    def _forward_remove_homologous_ts_edges(self, u, v, u_lag, v_lag):
        """Remove homologous time-series edges in the backwards direction."""
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)
        if u_lag <= v_lag:
            raise RuntimeError(f"From lag {u_lag} should be larger than to lag {v_lag}.")

        from_t = u_lag
        to_t = v_lag
        for _ in range(0, v_lag + 1):
            super().remove_edge((u, -from_t), (v, -to_t))
            from_t -= 1
            to_t -= 1


class StationaryTimeSeriesGraph(TsGraphEdgeMixin, BaseTimeSeriesGraph):
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

    # whether to deal with homologous edges when adding/removing
    _auto_addition: bool = True
    _auto_removal: Optional[str] = "backwards"

    def __init__(
        self, incoming_graph_data=None, max_lag: int = 1, check_time_direction: bool = True, **attr
    ):
        attr.update(dict(max_lag=max_lag, check_time_direction=check_time_direction))
        super(StationaryTimeSeriesGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, **attr
        )


class StationaryTimeSeriesDiGraph(TsGraphEdgeMixin, BaseTimeSeriesDiGraph):
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

    # whether to deal with homologous edges when adding/removing
    _auto_addition: bool = True
    _auto_removal: Optional[str] = "backwards"

    def __init__(
        self, incoming_graph_data=None, max_lag: int = 1, check_time_direction: bool = True, **attr
    ):
        attr.update(dict(max_lag=max_lag, check_time_direction=check_time_direction))
        super(StationaryTimeSeriesDiGraph, self).__init__(
            incoming_graph_data=incoming_graph_data, **attr
        )


class StationaryTimeSeriesMixedEdgeGraph(BaseTimeSeriesMixedEdgeGraph):
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


def complete_ts_graph(
    variables,
    max_lag: int,
    include_contemporaneous: bool = True,
    create_using=StationaryTimeSeriesGraph,
) -> StationaryTimeSeriesDiGraph:
    G = create_using(max_lag=max_lag)

    # add all possible edges
    for u_node in variables:
        for v_node in variables:
            for u_lag in range(max_lag + 1):
                for v_lag in range(max_lag + 1):
                    if u_lag < v_lag:
                        continue
                    # skip contemporaneous edges if necessary
                    if not include_contemporaneous and u_lag == v_lag:
                        continue
                    # do not add self connections
                    if u_node == v_node and u_lag == v_lag:
                        continue
                    # do not add cyclicity
                    if u_lag == v_lag and (
                        G.has_edge((u_node, -u_lag), (v_node, -v_lag))
                        or G.has_edge((v_node, -v_lag), (u_node, -u_lag))
                    ):
                        continue
                    # if there is already an edge, do not add
                    if G.has_edge((u_node, -u_lag), (v_node, -v_lag)):
                        continue

                    G.add_edge((u_node, -u_lag), (v_node, -v_lag))
    return G


def empty_ts_graph(
    variables, max_lag, create_using=StationaryTimeSeriesGraph
) -> StationaryTimeSeriesDiGraph:
    G = create_using(max_lag=max_lag)
    for node in variables:
        G.add_node((node, 0))
    return G


def nodes_in_time_order(G: BaseTimeSeriesGraph) -> Iterator:
    """Return nodes from G in time order starting from max-lag to t=0."""
    for t in range(G.max_lag, -1, -1):
        for node in G.nodes_at(t):
            yield node
