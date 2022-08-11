import networkx as nx
from graphs import MixedEdgeGraph

from .base import AncestralMixin


class ADMG(MixedEdgeGraph, AncestralMixin):
    """Acyclic directed mixed graph (ADMG).

    A causal graph with two different edge types: bidirected and traditional
    directed edges. Directed edges constitute causal relations as a
    causal DAG did, while bidirected edges constitute the presence of a
    latent confounder.

    Parameters
    ----------
    incoming_directed_edges : input directed edges (optional, default: None)
        Data to initialize directed edges. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    incoming_bidirected_edges : input bidirected edges (optional, default: None)
        Data to initialize bidirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    incoming_undirected_edges : input undirected edges (optional, default: None)
        Data to initialize undirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    directed_edge_name : str
        The name for the directed edges. By default 'directed'.
    bidirected_edge_name : str
        The name for the bidirected edges. By default 'bidirected'.
    undirected_edge_name : str
        The name for the directed edges. By default 'undirected'.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph
    MixedEdgeGraph

    Notes
    -----
    The data structure underneath the hood is stored in two networkx graphs:
    ``networkx.Graph`` and ``networkx.DiGraph`` to represent the non-directed
    edges and directed edges. Non-directed edges in an ADMG can be present as
    bidirected edges standing for latent confounders, or undirected edges
    standing for selection bias.

    - Normal directed edges (<-, ->, indicating causal relationship) = `networkx.DiGraph`
    - Bidirected edges (<->, indicating latent confounder) = `networkx.Graph`
    - Undirected edges (--, indicating selection bias) = `networkx.Graph`
    """

    def __init__(
        self,
        incoming_directed_edges=None,
        incoming_bidirected_edges=None,
        incoming_undirected_edges=None,
        directed_edge_name="directed",
        bidirected_edge_name="bidirected",
        undirected_edge_name="undirected",
        **attr,
    ):
        super().__init__(**attr)
        self.add_edge_type(nx.DiGraph(incoming_directed_edges), directed_edge_name)
        self.add_edge_type(nx.Graph(incoming_bidirected_edges), bidirected_edge_name)
        self.add_edge_type(nx.Graph(incoming_undirected_edges), undirected_edge_name)

        self._directed_name = directed_edge_name
        self._bidirected_name = bidirected_edge_name
        self._undirected_name = undirected_edge_name

        if not nx.is_directed_acyclic_graph(self.sub_directed_graph()):
            raise RuntimeError(f"{self} is not a DAG, which it should be.")

    @property
    def undirected_edge_name(self):
        return self._undirected_name

    @property
    def directed_edge_name(self):
        return self._directed_name

    @property
    def bidirected_edge_name(self):
        return self._bidirected_name

    def c_components(self):
        """Generate confounded components of the graph.

        Returns
        -------
        comp : iterator of sets
            The c-components.
        """
        return nx.connected_components(self.sub_bidirected_graph())
        # return [comp for comp in c_comps if len(comp) > 1]

    @property
    def bidirected_edges(self) -> nx.reportviews.EdgeView:
        """`EdgeView` of the bidirected edges."""
        return self.get_graphs(self._bidirected_name).edges

    @property
    def undirected_edges(self) -> nx.reportviews.EdgeView:
        """`EdgeView` of the undirected edges."""
        return self.get_graphs(self._undirected_name).edges

    @property
    def directed_edges(self) -> nx.reportviews.EdgeView:
        """`EdgeView` of the directed edges."""
        return self.get_graphs(self._directed_name).edges

    def sub_directed_graph(self) -> nx.DiGraph:
        """Sub-graph of just the directed edges."""
        return self._get_internal_graph(self._directed_name)

    def sub_bidirected_graph(self) -> nx.Graph:
        """Sub-graph of just the bidirected edges."""
        return self._get_internal_graph(self._bidirected_name)

    def sub_undirected_graph(self) -> nx.Graph:
        """Sub-graph of just the undirected edges."""
        return self._get_internal_graph(self._undirected_name)
