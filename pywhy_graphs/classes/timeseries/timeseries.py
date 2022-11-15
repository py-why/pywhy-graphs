"""API for networkx-compliant time-series graphs.

"""
from copy import copy
from typing import Dict, Iterator, List, Optional, Set

import networkx as nx
import numpy as np
from networkx import NetworkXError
from networkx.classes.graph import _CachedPropertyResetterAdj

from pywhy_graphs.typing import Node

from .base import BaseTimeSeriesDiGraph, BaseTimeSeriesGraph, BaseTimeSeriesMixedEdgeGraph


class TsGraphNodeMixin:
    """A mixin class for dealing with nodes in time-series graph."""

    nodes: Iterator  # should be defined in parent class

    def _check_ts_node(self, node):
        if not isinstance(node, tuple) or len(node) != 2:
            raise ValueError(
                f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
                f"You passed in {node}."
            )
        if node[1] > 0:
            raise ValueError(f"All lag points should be 0, or less. You passed in {node}.")

        # Note: this uses the public max-lag, since the user should not be exposed to the
        # private max
        if node[1] < -self.max_lag:
            raise ValueError(f"Lag {node[1]} cannot be greater than set max_lag {self.max_lag}.")

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

    def nodes_at(self, t) -> Set:
        """Nodes at t=max_lag.

        Lag is a positive number, so a node at lag = 2,
        would be at time point "-2".
        """
        if t < 0:
            raise RuntimeError(f"Lag is a positive number. You passed {t}.")
        nodes = set()
        for node in self.nodes:
            if node[1] == -t:
                nodes.add(node)
        return nodes

    def add_node(self, node_name, **attr):
        """Add node in time."""
        self._check_ts_node(node_name)
        super().add_node(node_name, **attr)
        var_name, _ = node_name

        for t in range(self._max_lag + 1):
            super().add_node((var_name, -t), **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add nodes in time."""
        for node in nodes_for_adding:
            newdict = attr.copy()

            if isinstance(node[0], tuple):
                ndict = node[1]
                node = node[0]
                newdict = attr.copy()
                newdict.update(ndict)
            self.add_node(node, **newdict)

    def remove_node(self, node_name):
        """Remove node in time."""
        self._check_ts_node(node_name)
        var_name, _ = node_name
        super().remove_node(node_name)
        if self._auto_removal is not None:
            for t in range(self._max_lag + 1):
                try:
                    super().remove_node((var_name, -t))
                except NetworkXError:
                    continue

    def remove_nodes_from(self, ebunch):
        """Remove nodes in time."""
        for node in ebunch:
            self.remove_node(node)


class TsGraphPropertyMixin:
    """A mixin class for time-series graph properties."""

    graph: Dict
    _adj: _CachedPropertyResetterAdj

    @property
    def max_lag(self) -> int:
        """The maximum time-index lag."""
        return self._max_lag

    @property
    def _max_lag(self) -> int:
        """Private property to query the maximum time-index lag stored in the graph.

        In stationary graphs, this is 2 times the maximum lag to enable proper
        d-separation querying.
        """
        return self.graph["max_lag"]

    def set_auto_removal(self, auto: Optional[str]):
        AUTO_KEYWORDS = ["backwards", "forwards", None]
        if auto not in AUTO_KEYWORDS:
            raise ValueError(f"Auto removal should be one of {AUTO_KEYWORDS}.")
        self._auto_removal = auto

    def copy(self, as_view: bool = False, double_max_lag: bool = True):
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

        if double_max_lag:
            G.graph["max_lag"] = G.max_lag * 2

        G.add_nodes_from((n, d.copy()) for n, d in self._node.items() if n[1] == 0)  # type: ignore
        G.add_edges_from(  # type: ignore
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
            if v[1] == 0
        )

        # prevent auto-removal of homologous nodes and edges
        G.set_auto_removal(None)
        return G


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

    def lagged_neighbors(self, u):
        """Neighbors from t < u's current time index."""
        nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] < 0]

    def contemporaneous_neighbors(self, u):
        """Neighbors from the same time index as u."""
        nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] == 0]

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if u_lag > 0 or v_lag > 0:
            raise RuntimeError(f"All lags should be negative or 0, not {u_lag} or {v_lag}.")
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
            if u_lag < 0:
                # remove all homologous edges
                if self._auto_removal == "backwards":
                    self._backward_remove_homologous_ts_edges(u, v, u_lag, v_lag)
                elif self._auto_removal == "forwards":
                    self._forward_remove_homologous_ts_edges(u, v, u_lag, v_lag)
            elif u_lag == v_lag:
                # remove all contemporaneous homologous edges
                if self._auto_removal == "backwards":
                    self._backward_remove_homologous_contemporaneous_edges(u, v)
                elif self._auto_removal == "forwards":
                    self._forward_remove_homologous_contemporaneous_edges(u, v, u_lag)
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

    def set_max_lag(self, lag: int) -> None:
        if lag <= 0:
            raise ValueError(
                f"Max lag must always be greater than 0, so passed in {lag} value is invalid."
            )
        max_lag = copy(self.max_lag)  # type: ignore
        self.graph["max_lag"] = lag

        # we need to add edges
        if max_lag > lag:
            # get all non-lag nodes
            non_lag_nodes = self.nodes_at(t=0)  # type: ignore

            # now get all neighbors that are in the past
            edge_list = []
            for node in non_lag_nodes:
                edge_list.extend([(node, nbr) for nbr in self.lagged_neighbors(node)])

            # now add all homologous edges
            self.add_edges_from(edge_list)

        # here, we need to remove edges that are at higher lags
        elif max_lag < lag:
            for lag in range(max_lag, lag, -1):
                # get all non-lag nodes
                nodes = self.nodes_at(t=-lag)  # type: ignore
                self.remove_nodes_from(nodes)  # type: ignore


class StationaryTimeSeriesGraph(
    TsGraphNodeMixin, TsGraphPropertyMixin, TsGraphEdgeMixin, BaseTimeSeriesGraph
):
    """Stationary time-series graph without directionality on edges.

    This class in essence should not be used directly.

    Included for completeness to enable modeling and working with ``nx.Graph`` like
    objects with time-series structure. By the time-ordering assumption, all lagged
    edges must point forward in time.
    """

    # whether to deal with homologous edges when adding/removing
    _auto_addition: bool = True
    _auto_removal: Optional[str] = "backwards"

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
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


class StationaryTimeSeriesDiGraph(
    TsGraphNodeMixin, TsGraphPropertyMixin, TsGraphEdgeMixin, BaseTimeSeriesDiGraph
):
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
    """

    # whether to deal with homologous edges when adding/removing
    _auto_addition: bool = True
    _auto_removal: Optional[str] = "backwards"

    def __init__(self, incoming_graph_data=None, max_lag: int = 1, **attr):
        if max_lag <= 0:
            raise ValueError(f"Max lag for time series graph should be at least 1, not {max_lag}.")
        attr.update(dict(max_lag=max_lag))
        super(BaseTimeSeriesDiGraph, self).__init__(incoming_graph_data=None, **attr)
        if incoming_graph_data is not None:
            # we assume a list of tuples of tuples as edges
            if isinstance(incoming_graph_data, list):
                self.add_edges_from(incoming_graph_data)
            elif isinstance(incoming_graph_data, nx.Graph):
                self.add_edges_from(incoming_graph_data.edges)
            else:
                raise RuntimeError(
                    f"Not implemented yet for incoming graph data that is of "
                    f"type {type(incoming_graph_data)}."
                )


class StationaryTimeSeriesMixedEdgeGraph(
    TsGraphNodeMixin, TsGraphPropertyMixin, BaseTimeSeriesMixedEdgeGraph
):
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
