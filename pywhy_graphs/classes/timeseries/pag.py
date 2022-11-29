from typing import Dict, FrozenSet

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
        incoming_directed_edges=None,
        incoming_bidirected_edges=None,
        incoming_undirected_edges=None,
        directed_edge_name: str = "directed",
        bidirected_edge_name: str = "bidirected",
        undirected_edge_name: str = "undirected",
        **attr,
    ):
        super().__init__(**attr)
        self.add_edge_type(StationaryTimeSeriesDiGraph(incoming_directed_edges), directed_edge_name)
        self.add_edge_type(
            StationaryTimeSeriesGraph(incoming_undirected_edges), undirected_edge_name
        )
        self.add_edge_type(
            StationaryTimeSeriesGraph(incoming_bidirected_edges), bidirected_edge_name
        )

        self._directed_name = directed_edge_name
        self._undirected_name = undirected_edge_name
        self._bidirected_name = bidirected_edge_name
        from pywhy_graphs import is_valid_mec_graph

        # check that construction of PAG was valid
        is_valid_mec_graph(self)

        # extended patterns store unfaithful triples
        # these can be used for conservative structure learning algorithm
        self._unfaithful_triples: Dict[FrozenSet[Node], None] = dict()
