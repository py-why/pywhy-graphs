from typing import Dict, FrozenSet, Iterator, Protocol, Set

import networkx as nx

from pywhy_graphs.typing import Node


class GraphMixinProtocol(Protocol):
    """Protocol for any mixin for graphs."""

    def sub_directed_graph(self) -> nx.DiGraph:
        """Sub-graph over directed edges."""
        pass


class ConservativeMixinProtocol(Protocol):
    """Protocol for any mixin for conservative graphs."""

    _unfaithful_triples: Dict[FrozenSet, None]

    @property
    def nodes(self) -> nx.reportviews.NodeView:
        """Nodes of a graph."""
        pass


class AncestralMixin(GraphMixinProtocol):
    """Mixin for graphs with ancestral functions.

    Requires the inheriting class to define a sub-graph over directed
    edges.
    """

    def predecessors(self, source: Node) -> Set:
        """Ancestors of 'source' node with directed path."""
        return nx.ancestors(self.sub_directed_graph(), source)  # type: ignore

    def successors(self, source: Node) -> Set:
        """Descendants of 'source' node with directed path."""
        return nx.descendants(self.sub_directed_graph(), source)  # type: ignore

    def children(self, n: Node) -> Iterator:
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
        return self.sub_directed_graph().successors(n)  # type: ignore

    def parents(self, n: Node) -> Iterator:
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
        return self.sub_directed_graph().predecessors(n)  # type: ignore


class ConservativeMixin(ConservativeMixinProtocol):
    """Mixin for conservative graphs."""

    def mark_unfaithful_triple(self, v_i: Node, u: Node, v_j: Node) -> None:
        """Mark an unfaithful triple.

        Parameters
        ----------
        v_i : node
            The first node in a triple.
        u : node
            The second node in a triple.
        v_j : node
            The third node in a triple.
        """
        if any(node not in self.nodes for node in [v_i, u, v_j]):
            raise RuntimeError(f"The triple {v_i}, {u}, {v_j} is not in the graph.")

        self._unfaithful_triples[frozenset(v_i, u, v_j)] = None

    @property
    def excluded_triples(self) -> Dict[FrozenSet, None]:
        """Unfaithful triples."""
        return self._unfaithful_triples
