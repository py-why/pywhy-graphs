import numpy as np

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.typing import TsNode

from .base import BaseTimeSeriesGraph, tsdict
from .timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    TimeSeriesDiGraph,
    TimeSeriesGraph,
)


class TimeSeriesMixedEdgeGraph(BaseTimeSeriesGraph, pywhy_nx.MixedEdgeGraph):
    """A class to imbue mixed-edge graph with time-series structure.

    This should not be used directly.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = False

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict

    # supported graph types
    graph_types = (TimeSeriesGraph, TimeSeriesDiGraph)

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
            issubclass(graph.__class__, self.graph_types) for graph in graphs
        ):
            raise RuntimeError("All graphs for timeseries mixed-edge graph")

        attr.update(dict(max_lag=max_lag))
        self.graph = dict()
        self.graph["max_lag"] = max_lag
        super().__init__(graphs, edge_types, **attr)

    def copy(self):
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
        :meth:`pywhy_graphs.networkx.MixedEdgeGraph.to_directed`: return a
            directed copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        G = self.__class__(max_lag=self.max_lag)
        G.graph.update(self.graph)
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
        return G

    def add_edge(self, u_of_edge: TsNode, v_of_edge: TsNode, edge_type: str = "all", **attr):
        super().add_edge(u_of_edge, v_of_edge, edge_type=edge_type, **attr)

    def add_edges_from(self, ebunch, edge_type="all", **attr):
        super().add_edges_from(ebunch, edge_type=edge_type, **attr)

    def remove_edge(self, u_of_edge, v_of_edge, edge_type="all"):
        super().remove_edge(u_of_edge, v_of_edge, edge_type)  # type: ignore

    def remove_edges_from(self, ebunch, edge_type="all"):
        for edge in ebunch:
            self.remove_edge(*edge, edge_type)

    def add_homologous_edges(self, u_of_edge: TsNode, v_of_edge: TsNode, direction="both", **attr):
        """Add homologous edges.

        Assumes the edge that we consider is ``(u_of_edge, v_of_edge)``, that is 'u' points to 'v'.

        Parameters
        ----------
        u_of_edge : TsNode
            The from node.
        v_of_edge : TsNode
            The to node. The absolute value of the time lag should be less than or equal to
            the from node's time lag.
        direction : str, optional
            Which direction to add homologous edges to, by default 'both', corresponding
            to making the edge stationary over all time.
        """
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)

        u, u_lag = u_of_edge
        v, v_lag = v_of_edge

        # take absolute value
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        if direction == "both":
            # re-center to 0, assuming v_lag is smaller, since it is the "to node"
            u_lag = u_lag - v_lag
            v_lag = 0

            # now add lagged edges up until max lag
            to_t = v_lag
            from_t = u_lag
            for _ in range(u_lag, self._max_lag + 1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t += 1
                from_t += 1
        elif direction == "forward":
            # decrease lag moving forward
            for _ in range(v_lag, -1, -1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t -= 1
                from_t -= 1
        elif direction == "backwards":
            for _ in range(u_lag, self._max_lag + 1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t += 1
                from_t += 1

    def remove_homologous_edges(self, u_of_edge: TsNode, v_of_edge: TsNode, direction="both"):
        """Remove homologous edges.

        Assumes the edge that we consider is ``(u_of_edge, v_of_edge)``, that is 'u' points to 'v'.

        Parameters
        ----------
        u_of_edge : TsNode
            The from node.
        v_of_edge : TsNode
            The to node. The absolute value of the time lag should be less than or equal to
            the from node's time lag.
        direction : str, optional
            Which direction to add homologous edges to, by default 'both', corresponding
            to making the edge stationary over all time.
        """
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)

        u, u_lag = u_of_edge
        v, v_lag = v_of_edge

        # take absolute value
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        if direction == "both":
            # re-center to 0, assuming v_lag is smaller, since it is the "to node"
            u_lag = u_lag - v_lag
            v_lag = 0

            # now add lagged edges up until max lag
            to_t = v_lag
            from_t = u_lag
            for _ in range(u_lag, self._max_lag + 1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t += 1
                from_t += 1
        elif direction == "forward":
            to_t = v_lag
            from_t = u_lag
            # decrease lag moving forward
            for _ in range(v_lag, -1, -1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t -= 1
                from_t -= 1
        elif direction == "backwards":
            to_t = v_lag
            from_t = u_lag
            for _ in range(u_lag, self._max_lag + 1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t += 1
                from_t += 1


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

    # supported graph types
    graph_types = (StationaryTimeSeriesGraph, StationaryTimeSeriesDiGraph)

    def __init__(self, graphs=None, edge_types=None, max_lag: int = None, **attr):
        # attr.update(dict(max_lag=max_lag))
        super().__init__(graphs, edge_types, max_lag=max_lag, **attr)
