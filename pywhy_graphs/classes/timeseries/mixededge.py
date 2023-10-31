from typing import Optional

import numpy as np

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.typing import TsNode

from .base import BaseTimeSeriesGraph, tsdict
from .digraph import StationaryTimeSeriesDiGraph, TimeSeriesDiGraph
from .graph import StationaryTimeSeriesGraph, TimeSeriesGraph


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

        Exactly the same as :meth:`pywhy_graphs.networkx.MixedEdgeGraph.copy`,
        except this preserves the max lag graph attribute.

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

    def add_homologous_edges(
        self, u_of_edge: TsNode, v_of_edge: TsNode, direction="both", edge_type="all", **attr
    ):
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
        if edge_type == "all":
            for edge_type, graph in self.get_graphs().items():
                graph.add_homologous_edges(u_of_edge, v_of_edge, direction=direction, **attr)
        else:
            graph = self.get_graphs(edge_type=edge_type)
            graph.add_homologous_edges(u_of_edge, v_of_edge, direction=direction, **attr)

    def remove_homologous_edges(
        self, u_of_edge: TsNode, v_of_edge: TsNode, edge_type: str = "all", direction="both"
    ):
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
        if edge_type == "all":
            for edge_type, graph in self.get_graphs().items():
                graph.remove_homologous_edges(u_of_edge, v_of_edge, direction=direction)
        else:
            graph = self.get_graphs(edge_type=edge_type)
            graph.remove_homologous_edges(u_of_edge, v_of_edge, direction=direction)


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

    # whether or not the graph should be assumed to be stationary
    stationary: bool = True

    # supported graph types
    graph_types = (StationaryTimeSeriesGraph, StationaryTimeSeriesDiGraph)

    def __init__(self, graphs=None, edge_types=None, max_lag: Optional[int] = None, **attr):
        super().__init__(graphs, edge_types, max_lag=max_lag, **attr)

    def set_stationarity(self, stationary: bool):
        self.stationary = stationary
        for graph in self.get_graphs().values():
            graph.stationary = stationary
