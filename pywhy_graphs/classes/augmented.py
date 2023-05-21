import collections
from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Tuple

from networkx.classes.reportviews import NodeView

from pywhy_graphs.typing import Node

from .admg import ADMG
from .pag import PAG


class AugmentedNodeMixin:
    graph: dict
    nodes: NodeView

    @abstractmethod
    def add_edge(self, u_of_edge, v_of_edge, edge_type="all", **attr):
        pass

    @abstractmethod
    def add_node(self, u):
        pass

    @property
    @abstractmethod
    def directed_edge_name(self) -> str:
        pass

    def _verify_augmentednode_dict(self):
        # verify validity of F nodes
        if "F-nodes" not in self.graph:
            self.graph["F-nodes"] = collections.defaultdict(lambda: collections.defaultdict(set))
        elif not isinstance(self.graph["F-nodes"], dict):
            raise RuntimeError(
                "There is a graph property named F-nodes already that is not of type dict."
            )

        if "S-nodes" not in self.graph:
            self.graph["S-nodes"] = dict()
        elif not isinstance(self.graph["S-nodes"], dict):
            raise RuntimeError(
                "There is a graph property named S-nodes already that is not of type dict."
            )

    def add_f_node(self, intervention_set: Set[Node], require_unique=True, domain=None):
        """Add an F-node to the graph.

        Parameters
        ----------
        intervention_set : Set[Node]
            A set of regular nodes that already exist in the causal graph.
        require_unique : bool, optional
            Whether or not to require that the intervention set is unique. If False,
            then the intervention set is added to the graph, even if it is already
            an F-node. The default is True.
        domain : Optional[Set[int]], optional
            The domain of the F-node. If None, then the domain is just set to 1.
        """
        if isinstance(intervention_set, str) or not isinstance(intervention_set, Iterable):
            raise RuntimeError("The intervention set nodes must be an iterable set of node(s).")
        if domain is None:
            domain = set([1])

        # check that there are no duplicates and perform set conversion
        orig_len = len(intervention_set)
        intervention_set = frozenset(intervention_set)  # type: ignore
        if len(intervention_set) != orig_len:
            raise RuntimeError("The intervention set must be a set of unique nodes.")

        # check that the F-node intervention set has variables within the graph
        if require_unique and intervention_set in self.intervention_sets:
            raise RuntimeError(
                f"You cannot add an F-node for {intervention_set} because "
                f"there is already an F-node."
            )
        for node in intervention_set:
            if node not in self.nodes:
                raise RuntimeError(
                    f"All intervention sets must be nodes already in the graph. {node} is not."
                )

        # add a new F-node into the graph
        f_node_name = ("F", len(self.f_nodes))
        self.add_node(f_node_name)

        # add edge between the F-node and its intervention set
        for intervened_node in intervention_set:
            self.add_edge(f_node_name, intervened_node, self.directed_edge_name)

        # adding nodes to F-node container occurs last, because of the error checks
        # that occur in adding edges
        self.graph["F-nodes"][f_node_name]["targets"] = intervention_set
        self.graph["F-nodes"][f_node_name]["domain"] = domain

    def add_f_nodes_from(self, intervention_sets: List[Set[Node]]):
        """Add a bunch of F-nodes at once."""
        for intervention_set in intervention_sets:
            self.add_f_node(intervention_set)

    def set_f_node(self, f_node, targets: Optional[Set] = None):
        if f_node not in self.nodes:
            raise RuntimeError(f"{f_node} is not a node in the existing graph.")

        if targets is not None and not all(target in self.nodes for target in targets):
            raise RuntimeError(f"Not all targets {targets} are in the existing graph.")

        self.graph["F-nodes"][f_node]["targets"] = targets

    @property
    def augmented_nodes(self):
        """Return set of augmented nodes."""
        return self.f_nodes + self.s_nodes

    @property
    def f_nodes(self) -> List[Node]:
        """Return set of F-nodes."""
        return list(self.graph["F-nodes"].keys())

    @property
    def non_augmented_nodes(self):
        """Return set of non augmented-nodes."""
        return set(self.nodes).difference(self.f_nodes).difference(self.s_nodes)

    @property
    def intervention_sets(self):
        """Return set of intervention-sets."""
        targets = set()
        for f_node in self.f_nodes:
            targets.add(self.graph["F-nodes"][f_node]["targets"])
        return targets

    @property
    def intervened_nodes(self):
        """Return set of intervened nodes."""
        nodes = set()
        for iset in self.intervention_sets:
            nodes = nodes.union(iset)
        return nodes

    @property
    def domain_ids(self):
        """Return set of domain ids."""
        domain_ids = set()
        for src, target in self.graph["S-nodes"].values():
            domain_ids.add(src)
            domain_ids.add(target)

        return list(domain_ids)

    @property
    def s_nodes(self) -> List[Node]:
        """Return set of S-nodes."""
        return list(self.graph["S-nodes"].keys())

    def add_s_node(self, domain_ids: Tuple, node_changes: Set[Node] = None):
        if isinstance(node_changes, str) or not isinstance(node_changes, Iterable):
            raise RuntimeError("The intervention set nodes must be an iterable set of node(s).")

        # check that there are no duplicates and perform set conversion
        orig_len = len(node_changes)
        node_changes = frozenset(node_changes)  # type: ignore
        if len(node_changes) != orig_len:
            raise RuntimeError("The set must be a set of unique nodes.")

        # check that the F-node intervention set has variables within the graph
        if domain_ids in self.domain_ids:
            raise RuntimeError(
                f"You cannot add an augmneted-node for {node_changes} because "
                f"there is already an augmented-node."
            )

        # add a new S-node into the graph
        s_node_name = ("S", len(self.s_nodes))
        self.add_node(s_node_name, domain_ids=domain_ids)

        # add edge between the F-node and its intervention set
        for perturbed_node in node_changes:
            self.add_edge(s_node_name, perturbed_node, self.directed_edge_name)

        # adding nodes to F-node container occurs last, because of the error checks
        # that occur in adding edges
        self.graph["S-nodes"][s_node_name] = domain_ids


class AugmentedGraph(ADMG, AugmentedNodeMixin):
    """An augmented causal diagram.

    An augmented graph is one where interventions are represented by F-nodes.
    See :footcite:`pearl_aspects_1993`, where they were first introduced. They
    allow one to model hard and soft interventions as an explicit "F-node" added
    to the existing causal graph. For more information, see <TBD user guide>.

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
    ADMG
    pywhy_graphs.networkx.MixedEdgeGraph

    Notes
    -----
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

    def __init__(
        self,
        incoming_directed_edges=None,
        incoming_bidirected_edges=None,
        incoming_undirected_edges=None,
        directed_edge_name: str = "directed",
        bidirected_edge_name: str = "bidirected",
        undirected_edge_name: str = "undirected",
        **attr,
    ):
        super().__init__(
            incoming_directed_edges,
            incoming_bidirected_edges,
            incoming_undirected_edges,
            directed_edge_name,
            bidirected_edge_name,
            undirected_edge_name,
            **attr,
        )

        # verify validity of F nodes
        self._verify_augmentednode_dict()

    def remove_node(self, n):
        if n in self.f_nodes:
            del self.graph["F-nodes"][n]
        return super().remove_node(n)


class AugmentedPAG(PAG, AugmentedNodeMixin):
    """An augmented PAG.

    An augmented PAG is a PAG that has been augmented with either F-nodes or
    S-nodes, or both. It is a Markov equivalence class of causal diagrams.

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
    index pair of intervention-targets.

    If the intervention targets are unknown, then the 2-tuple contains
    integer indices representing the index of an interventional distribution.
    This is called :math:`\\sigma` in :footcite:`Jaber2020causal`.

    **Edge Type Subgraphs**

    Different edge types in an AugmentedPAG are represented exactly as they are in a
    :class:`pywhy_graphs.PAG`.

    **F-nodes**

    Interventions are represented by special nodes, known as F-nodes. See
    :footcite:`Jaber2020causal`, or :footcite:`Kocaoglu2019characterization` for details.

    F-nodes are represented in pywhy-graphs as a tuple as ``('F', <index>)``, where ``index``
    is just a random index number. Each F-node is mapped to the intervention-set that they
    are applied on. For example in the graph :math:`('F', 0) \\rightarrow X \\rightarrow Y`,
    ``('F', 0)`` is the F-node added that models an intervention on ``X``. Each intervention-set
    is a set of regular nodes in the causal graph.

    **S-nodes**

    Different domains and environments are represented by special nodes, known as S-nodes. See
    :footcite:`bareinboim_causal_2016` for details.

    S-nodes are represented in pywhy-graphs as a tuple as ``('S', <index>)``, where ``index``
    is just a random index number. Each F-node is mapped to the intervention-set that they
    are applied on. For example in the graph :math:`('F', 0) \\rightarrow X \\rightarrow Y`,
    ``('F', 0)`` is the F-node added that models an intervention on ``X``. Each intervention-set
    is a set of regular nodes in the causal graph.

    References
    ----------
    .. footbibliography::
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
        if n in self.s_nodes:
            del self.graph["S-nodes"][n]
        return super().remove_node(n)
