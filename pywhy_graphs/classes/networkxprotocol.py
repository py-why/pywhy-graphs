from abc import abstractmethod
from typing import Dict, Protocol


class NetworkXProtocol(Protocol):
    """A protocol to allow mypy type checking to pass."""

    graph: Dict
    _node: Dict
    _adj: Dict

    @property
    def nodes(self): ...

    @property
    def edges(self): ...

    @abstractmethod
    def add_node(self, node): ...

    @abstractmethod
    def remove_edge(self, u, v): ...
