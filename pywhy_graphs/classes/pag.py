from typing import Dict, FrozenSet, Iterator, Mapping

import networkx as nx

from ..typing import Node
from .admg import ADMG
from .base import ConservativeMixin


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
        self._unfaithful_triples: Dict[FrozenSet[Node], None] = dict()

        # check that construction of PAG was valid
        from pywhy_graphs import is_valid_mec_graph

        is_valid_mec_graph(self)

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

        # Performs orientation of edges
        if self.has_edge(v, u, self.directed_edge_name):
            # Orients: u <-o v => u <-> v
            # when we orient (u,v) now as an arrowhead, it is a bidirected arrow
            self.remove_edge(v, u, self.directed_edge_name)
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.bidirected_edge_name)
        elif self.has_edge(v, u, self.circle_edge_name):
            # Orients: u o-o v => u o-> v
            # In this case, we have a bidirected circle edge
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.directed_edge_name)
        elif self.has_edge(u, v, self.circle_edge_name):
            # In this case, we have a circle edge that is oriented into an arrowhead
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.directed_edge_name)
        else:  # noqa
            raise RuntimeError("The current PAG is invalid.")

    def possible_children(self, n: Node) -> Iterator:
        """Return an iterator over children of node n.

        Possible children of 'n' are nodes with an edge like
        'n' o-> 'x'. Nodes with 'n' o-o 'x' are not considered
        possible children.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        possible_children : Iterator
            An iterator of the children of node 'n'.
        """
        return self.sub_directed_graph().successors(n)

    def possible_parents(self, n: Node) -> Iterator:
        """Return an iterator over possible parents of node n.

        Possible parents of 'n' are nodes with an edge like
        'n' <-o 'x'. Nodes with 'n' o-o 'x' are not considered
        possible parents.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Returns
        -------
        possible_parents : Iterator
            An iterator of the parents of node 'n'.
        """
        return self.sub_directed_graph().predecessors(n)

    def parents(self, n: Node) -> Iterator:
        """Return the definite parents of node 'n' in a PAG.

        Definite parents are parents of node 'n' with only
        a directed edge between them from 'n' <- 'x'. For example,
        'n' <-o 'x' does not qualify 'x' as a parent of 'n'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Yields
        ------
        parents : Iterator
            An iterator of the definite parents of node 'n'.

        See Also
        --------
        possible_children
        children
        possible_parents
        """
        possible_parents = self.possible_parents(n)
        for node in possible_parents:
            if not self.has_edge(n, node, self.circle_edge_name):
                yield node

    def children(self, n: Node) -> Iterator:
        """Return the definite children of node 'n' in a PAG.

        Definite children are children of node 'n' with only
        a directed edge between them from 'n' -> 'x'. For example,
        'n' o-> 'x' does not qualify 'x' as a children of 'n'.

        Parameters
        ----------
        n : node
            A node in the causal DAG.

        Yields
        ------
        children : Iterator
            An iterator of the children of node 'n'.

        See Also
        --------
        possible_children
        parents
        possible_parents
        """
        possible_children = self.possible_children(n)
        for node in possible_children:
            if not self.has_edge(node, n, self.circle_edge_name):
                yield node

    def add_edge(self, u_of_edge, v_of_edge, edge_type="all", **attr):
        from pywhy_graphs.algorithms.generic import _check_adding_pag_edge

        _check_adding_pag_edge(self, u_of_edge=u_of_edge, v_of_edge=v_of_edge, edge_type=edge_type)
        return super().add_edge(u_of_edge, v_of_edge, edge_type, **attr)

    def add_edges_from(self, ebunch_to_add, edge_type, **attr):
        from pywhy_graphs.algorithms.generic import _check_adding_pag_edge

        for u_of_edge, v_of_edge in ebunch_to_add:
            _check_adding_pag_edge(
                self, u_of_edge=u_of_edge, v_of_edge=v_of_edge, edge_type=edge_type
            )
        return super().add_edges_from(ebunch_to_add, edge_type, **attr)
