from typing import Dict, Protocol


class NetworkXProtocol(Protocol):
    """A protocol to allow mypy type checking to pass."""

    graph: Dict
    _node: Dict
    _adj: Dict

    @property
    def nodes(self):
        pass

    @property
    def edges(self):
        pass

    def add_node(self, node):
        pass

    def remove_edge(self, u, v):
        pass
