from copy import deepcopy
from typing import Dict, FrozenSet, Iterator, Mapping

import networkx as nx

from pywhy_graphs.classes.base import AncestralMixin, ConservativeMixin
from pywhy_graphs.typing import Node

from .timeseries import (
    StationaryTimeSeriesDiGraph,
    StationaryTimeSeriesGraph,
    StationaryTimeSeriesMixedEdgeGraph,
)


class StationaryTimeSeriesPAG(
    StationaryTimeSeriesMixedEdgeGraph, AncestralMixin, ConservativeMixin
):
    def __init__(
        self,
        incoming_circle_edges=None,
        incoming_directed_edges=None,
        incoming_bidirected_edges=None,
        incoming_undirected_edges=None,
        circle_edge_name: str = "circle",
        directed_edge_name: str = "directed",
        bidirected_edge_name: str = "bidirected",
        undirected_edge_name: str = "undirected",
        **attr,
    ):
        super().__init__(**attr)
        self.add_edge_type(StationaryTimeSeriesDiGraph(incoming_directed_edges), directed_edge_name)
        self.add_edge_type(
            StationaryTimeSeriesDiGraph(incoming_circle_edges, check_time_direction=False),
            circle_edge_name,
        )
        self.add_edge_type(
            StationaryTimeSeriesGraph(incoming_undirected_edges), undirected_edge_name
        )
        self.add_edge_type(
            StationaryTimeSeriesGraph(incoming_bidirected_edges), bidirected_edge_name
        )

        self._directed_name = directed_edge_name
        self._undirected_name = undirected_edge_name
        self._circle_name = circle_edge_name
        self._bidirected_name = bidirected_edge_name
        from pywhy_graphs import is_valid_mec_graph

        # check that construction of PAG was valid
        is_valid_mec_graph(self)

        # extended patterns store unfaithful triples
        # these can be used for conservative structure learning algorithm
        self._unfaithful_triples: Dict[FrozenSet[Node], None] = dict()

    @property
    def undirected_edge_name(self) -> str:
        """Name of the undirected edge internal graph."""
        return self._undirected_name

    @property
    def directed_edge_name(self) -> str:
        """Name of the directed edge internal graph."""
        return self._directed_name

    @property
    def bidirected_edge_name(self) -> str:
        """Name of the bidirected edge internal graph."""
        return self._bidirected_name

    @property
    def circle_edge_name(self) -> str:
        """Name of the bidirected edge internal graph."""
        return self._circle_name

    @property
    def undirected_edges(self) -> Mapping:
        """``EdgeView`` of the undirected edges."""
        return self.get_graphs(self._undirected_name).edges

    @property
    def bidirected_edges(self) -> Mapping:
        """``EdgeView`` of the bidirected edges."""
        return self.get_graphs(self._bidirected_name).edges

    @property
    def directed_edges(self) -> Mapping:
        """``EdgeView`` of the directed edges."""
        return self.get_graphs(self._directed_name).edges

    @property
    def circle_edges(self) -> Mapping:
        """``EdgeView`` of the directed edges."""
        return self.get_graphs(self.circle_edge_name).edges

    def sub_directed_graph(self) -> nx.DiGraph:
        """Sub-graph of just the directed edges."""
        return self._get_internal_graph(self._directed_name)

    def sub_undirected_graph(self) -> nx.Graph:
        """Sub-graph of just the undirected edges."""
        return self._get_internal_graph(self._undirected_name)

    def sub_bidirected_graph(self) -> nx.Graph:
        """Sub-graph of just the bidirected edges."""
        return self._get_internal_graph(self._bidirected_name)

    def sub_circle_graph(self) -> nx.Graph:
        """Sub-graph of just the circle edges."""
        return self._get_internal_graph(self.circle_edge_name)

    def orient_uncertain_edge(self, u: Node, v: Node) -> None:
        """Orient undirected edge into an arrowhead.

        If there is an undirected edge u - v, then the arrowhead
        will orient u -> v. If the correct order is v <- u, then
        simply pass the arguments in different order.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        """
        if not self.has_edge(u, v, self._undirected_name):
            raise RuntimeError(f"There is no undirected edge between {u} and {v}.")
        u, v = sorted([u, v], key=lambda x: x[1])  # type: ignore
        self.remove_edge(u, v, self.circle_edge_name)
        self.add_edge(u, v, self._directed_name)

    def possible_children(self, n: Node) -> Iterator[Node]:
        """Return an iterator over children of node n.

        Children of node 'n' are nodes with a directed
        edge from 'n' to that node. For example,
        'n' -> 'x', 'n' -> 'y'. Nodes only connected
        via a bidirected edge are not considered children:
        'n' <-> 'y'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        children : Iterator
            An iterator of the children of node 'n'.
        """
        return self.sub_undirected_graph().neighbors(n)

    def possible_parents(self, n: Node) -> Iterator[Node]:
        """Return an iterator over parents of node n.

        Parents of node 'n' are nodes with a directed
        edge from 'n' to that node. For example,
        'n' <- 'x', 'n' <- 'y'. Nodes only connected
        via a bidirected edge are not considered parents:
        'n' <-> 'y'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        parents : Iterator
            An iterator of the parents of node 'n'.
        """
        return self.sub_undirected_graph().neighbors(n)

    def to_ts_undirected(self):
        graph_class = StationaryTimeSeriesGraph

        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, 0) for n in self.variables)
        G.add_edges_from(
            (u, v, deepcopy(d))
            for _, edge_adj in self.adj.items()
            for u, nbrs in edge_adj.items()
            for v, d in nbrs.items()
        )
        return G
