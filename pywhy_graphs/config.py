from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """Meta enumeration to make 'in' keyword work."""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

    # Prints out the name of the type
    def __str__(self):
        return self.name


class EdgeType(Enum, metaclass=MetaEnum):
    """Enumeration of different causal edge endpoints.

    Categories
    ----------
    directed : str
        Signifies arrowhead ("->") edges.
    circle : str
        Signifies a circle ("*-o") endpoint. That is an uncertain edge,
        which is either circle with directed edge (``o->``),
        circle with undirected edge (``o-``), or
        circle with circle edge (``o-o``).
    undirected : str
        Signifies an undirected ("-") edge. That is an undirected edge (``-``),
        or circle with circle edge (``-o``).

    Notes
    -----
    The possible edges between two nodes thus are:

    ->, <-, <->, o->, <-o, o-o

    In general, among all possible causal graphs, arrowheads depict
    non-descendant relationships. In DAGs, arrowheads depict direct
    causal relationships (i.e. parents/children). In ADMGs, arrowheads
    can come from directed edges, or bidirected edges
    """

    ALL = "all"
    DIRECTED = "directed"
    BIDIRECTED = "bidirected"
    CIRCLE = "circle"
    UNDIRECTED = "undirected"


# Taken from causal-learn Endpoint.py
# A typesafe enumeration of the types of endpoints that are permitted in
# Tetrad-style graphs: tail (--) null (-), arrow (->), circle (-o) and star (-*).
# 'TAIL_AND_ARROW' and 'ARROW_AND_ARROW' means there are two types of edges (<-> and -->)
# between two nodes.
# 'TAIL_AND_TAIL' means there are two types of edges with two tails ending on this endpoint
class CLearnEndpoint(Enum, metaclass=MetaEnum):
    """Enumeration of causal-learn endpoints."""

    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2
    STAR = 3
    TAIL_AND_ARROW = 4
    ARROW_AND_ARROW = 5
    TAIL_AND_TAIL = 6  # added by pywhy.
