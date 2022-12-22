from typing import Dict, FrozenSet, Iterator, Mapping

import networkx as nx

import pywhy_graphs.networkx as pywhy_nx

from ..typing import Node
from .base import AncestralMixin, ConservativeMixin


class CPDAG(pywhy_nx.MixedEdgeGraph, AncestralMixin, ConservativeMixin):
    """Completed partially directed acyclic graphs (CPDAG).

    CPDAGs generalize causal DAGs by allowing undirected edges.
    Undirected edges imply uncertainty in the orientation of the causal
    relationship. For example, ``A - B``, can be ``A -> B`` or ``A <- B``,
    allowing for a Markov equivalence class of DAGs for each CPDAG.

    Parameters
    ----------
    incoming_directed_edges : input directed edges (optional, default: None)
        Data to initialize directed edges. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    incoming_undirected_edges : input undirected edges (optional, default: None)
        Data to initialize undirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    directed_edge_name : str
        The name for the directed edges. By default 'directed'.
    undirected_edge_name : str
        The name for the directed edges. By default 'undirected'.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph
    ADMG

    Notes
    -----
    CPDAGs are Markov equivalence class of causal DAGs. The implicit assumption in
    these causal graphs are the Structural Causal Model (or SCM) is Markovian, inducing
    causal sufficiency, where there is no unobserved latent confounder. This allows CPDAGs
    to be learned from score-based (such as the "GES" algorithm) and constraint-based
    (such as the PC algorithm) approaches for causal structure learning.

    One should not use CPDAGs if they suspect their data has unobserved latent confounders.
    """

    def __init__(
        self,
        incoming_directed_edges=None,
        incoming_undirected_edges=None,
        directed_edge_name: str = "directed",
        undirected_edge_name: str = "undirected",
        **attr,
    ):
        super().__init__(**attr)
        self.add_edge_type(nx.DiGraph(incoming_directed_edges), directed_edge_name)
        self.add_edge_type(nx.Graph(incoming_undirected_edges), undirected_edge_name)

        self._directed_name = directed_edge_name
        self._undirected_name = undirected_edge_name

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
    def undirected_edges(self) -> Mapping:
        """``EdgeView`` of the undirected edges."""
        return self.get_graphs(self._undirected_name).edges

    @property
    def directed_edges(self) -> Mapping:
        """``EdgeView`` of the directed edges."""
        return self.get_graphs(self._directed_name).edges

    def sub_directed_graph(self) -> nx.DiGraph:
        """Sub-graph of just the directed edges."""
        return self._get_internal_graph(self._directed_name)

    def sub_undirected_graph(self) -> nx.Graph:
        """Sub-graph of just the undirected edges."""
        return self._get_internal_graph(self._undirected_name)

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

        self.remove_edge(v, u, self._undirected_name)
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

    def add_edge(self, u_of_edge, v_of_edge, edge_type="all", **attr):
        from pywhy_graphs.algorithms.generic import _check_adding_cpdag_edge

        _check_adding_cpdag_edge(
            self, u_of_edge=u_of_edge, v_of_edge=v_of_edge, edge_type=edge_type
        )
        return super().add_edge(u_of_edge, v_of_edge, edge_type, **attr)

    def add_edges_from(self, ebunch_to_add, edge_type, **attr):
        from pywhy_graphs.algorithms.generic import _check_adding_cpdag_edge

        for u_of_edge, v_of_edge in ebunch_to_add:
            _check_adding_cpdag_edge(
                self, u_of_edge=u_of_edge, v_of_edge=v_of_edge, edge_type=edge_type
            )
        return super().add_edges_from(ebunch_to_add, edge_type, **attr)
