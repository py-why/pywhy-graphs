from typing import Iterator, Set

import networkx as nx


class AncestralMixin:
    """Mixin for graphs with ancestral functions."""

    def ancestors(self, source) -> Set:
        """Ancestors of 'source' node with directed path."""
        return nx.ancestors(self.sub_directed_graph(), source)  # type: ignore

    def descendants(self, source) -> Set:
        """Descendants of 'source' node with directed path."""
        return nx.descendants(self.sub_directed_graph(), source)  # type: ignore

    def children(self, n) -> Iterator:
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

    def parents(self, n) -> Iterator:
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
