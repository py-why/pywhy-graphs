from typing import Iterator, Mapping

import networkx as nx

from pywhy_graphs.base import ConservativeMixin
from pywhy_graphs.typing import Node

from .admg import ADMG


class PAG(ADMG, ConservativeMixin):
    """Partial ancestral graph (PAG).

    PAGs are a Markov equivalence class with mixed edges of directed,
    bidirected, undirected and edges with circle endpoints. In terms
    of graph functionality, they essentially extend the definition of
    an ADMG with edges with circular endpoints.

    Parameters
    ----------
    incoming_directed_edges : input directed edges (optional, default: None)
        Data to initialize directed edges. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    incoming_undirected_edges : input undirected edges (optional, default: None)
        Data to initialize undirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    incoming_bidirected_edges : input bidirected edges (optional, default: None)
        Data to initialize bidirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    incoming_circle_edges : input circular endpoint edges (optional, default: None)
        Data to initialize edges with circle endpoints. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    directed_edge_name : str
        The name for the directed edges. By default 'directed'.
    undirected_edge_name : str
        The name for the undirected edges. By default 'undirected'.
    bidirected_edge_name : str
        The name for the bidirected edges. By default 'bidirected'.
    circle_edge_name : str
        The name for the circle edges. By default 'circle'.
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
        incoming_bidirected_edges=None,
        incoming_circle_edges=None,
        directed_edge_name: str = "directed",
        undirected_edge_name: str = "undirected",
        bidirected_edge_name: str = "bidirected",
        circle_edge_name: str = "circle",
        **attr,
    ):
        super().__init__(
            incoming_directed_edges=incoming_directed_edges,
            incoming_undirected_edges=incoming_undirected_edges,
            incoming_bidirected_edges=incoming_bidirected_edges,
            directed_edge_name=directed_edge_name,
            undirected_edge_name=undirected_edge_name,
            bidirected_edge_name=bidirected_edge_name,
            **attr,
        )

        # add circular edges
        self.add_edge_type(nx.DiGraph(incoming_circle_edges), circle_edge_name)
        self._circle_name = circle_edge_name

        # extended patterns store unfaithful triples
        # these can be used for conservative structure learning algorithm
        self._unfaithful_triples = dict()

    @property
    def circle_edge_name(self) -> str:
        return self._circle_name

    @property
    def circle_edges(self) -> Mapping:
        """``EdgeView`` of the circle edges."""
        return self.get_graphs(self.circle_edge_name).edges

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
        if not self.has_edge(u, v, self.circle_edge_name):
            raise RuntimeError(f"There is no uncertain circular edge between {u} and {v}.")

        self.remove_edge(v, u, self.circle_edge_name)
        self.add_edge(u, v, self.directed_edge_name)

    def possible_children(self, n: Node) -> Iterator:
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
        return self.sub_circle_graph().neighbors(n)

    def possible_parents(self, n: Node) -> Iterator:
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
        return self.sub_circle_graph().neighbors(n)
