from .augmented import AugmentedNodeMixin
from .pag import PAG


class IPAG(PAG, AugmentedNodeMixin):
    """A I-PAG Markov equivalence class of causal graphs.

    A I-PAG is an equivalence class representing causal graphs that
    have known-target soft interventions associated with it. The interventions are
    represented by special nodes, known as F-nodes. See
    :footcite:`Kocaoglu2019characterization` for details.

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
    f_nodes : List[Node], optional
        List of corresponding nodes that are F nodes, by default None.

    See Also
    --------
    PsiPAG
    PAG

    Notes
    -----
    F-nodes are just nodes that are added to a causal graph, and
    represent an "augmentation" of the original causal graph to handle
    interventions. Each F-node is mapped to a 2-tuple representing the
    index pair of unknown interventions. Since intervention targets are known,
    then the 2-tuple contains nodes in the graph. This is called :math:`\\sigma`
    in :footcite:`Jaber2020causal`.

    Since F-nodes in an IPAG is defined by its source pair of interventions.

    **Edge Type Subgraphs**

    Different edge types in an I-PAG are represented exactly as they are in a
    :class:`pywhy_graphs.PAG`.

    **F-nodes**

    F-nodes are represented in pywhy-graphs as a tuple as ``('F', <index>)``, where ``index``
    is just a random index number. Each F-node is mapped to the intervention-set that they
    are applied on. For example in the graph :math:`('F', 0) \\rightarrow X \\rightarrow Y`,
    ``('F', 0)`` is the F-node added that models an intervention on ``X``. Each intervention-set
    is a set of regular nodes in the causal graph.

    References
    ----------
    .. footbibliography::
    """

    known_targets: bool = True

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
            incoming_directed_edges,
            incoming_undirected_edges,
            incoming_bidirected_edges,
            incoming_circle_edges,
            directed_edge_name,
            undirected_edge_name,
            bidirected_edge_name,
            circle_edge_name,
            **attr,
        )

        self._verify_augmentednode_dict()

    def remove_node(self, n):
        if n in self.f_nodes:
            del self.graph["F-nodes"][n]
        return super().remove_node(n)


class PsiPAG(PAG, AugmentedNodeMixin):
    """A Psi-PAG Markov equivalence class of causal graphs.

    A Psi-PAG is an equivalence class representing causal graphs that
    have unknown-target soft interventions associated with it. The interventions
    are represented by special nodes, known as F-nodes. See
    :footcite:`Jaber2020causal` for details.

    A PsiPAG is inherently different from a IPAG in that the F-nodes are not
    associated with any node because the interventions have "unknown targets".

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
    f_nodes : List[Node], optional
        List of corresponding nodes that are F nodes, by default None.

    Notes
    -----
    F-nodes are just nodes that are added to a causal graph, and
    represent an "augmentation" of the original causal graph to handle
    interventions. Each F-node is mapped to a 2-tuple representing the
    index pair of unknown interventions. Since intervention targets are unknown,
    then the 2-tuple contains integer indices representing the index of
    an interventional distribution. This is called :math:`\\sigma` in
    :footcite:`Jaber2020causal`.

    **Edge Type Subgraphs**

    Different edge types in an I-PAG are represented exactly as they are in a
    :class:`pywhy_graphs.PAG`.

    **F-nodes**

    F-nodes are represented in pywhy-graphs as a tuple as ``('F', <index>)``, where ``index``
    is just a random index number. Each F-node is mapped to the intervention-set that they
    are applied on. For example in the graph :math:`('F', 0) \\rightarrow X \\rightarrow Y`,
    ``('F', 0)`` is the F-node added that models an intervention on ``X``. Each intervention-set
    is a set of regular nodes in the causal graph.

    References
    ----------
    .. footbibliography::
    """

    known_targets: bool = False

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
            incoming_directed_edges,
            incoming_undirected_edges,
            incoming_bidirected_edges,
            incoming_circle_edges,
            directed_edge_name,
            undirected_edge_name,
            bidirected_edge_name,
            circle_edge_name,
            **attr,
        )

        self._verify_augmentednode_dict()

    def remove_node(self, n):
        if n in self.f_nodes:
            del self.graph["F-nodes"][n]
        return super().remove_node(n)
