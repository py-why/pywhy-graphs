from typing import Any, Tuple, Union

# Type hint for any "node" in a causal graph that is compliant with
# what is typically stored in pandas. Note that this is a limitation
# compared with what networkx Nodes allow.
Node = Union[int, float, str, Any]

TsNode = Tuple[Node, int]
