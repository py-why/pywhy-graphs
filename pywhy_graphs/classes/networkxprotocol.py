from typing import Dict, Protocol


class NetworkXProtcol(Protocol):
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
