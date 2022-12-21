from copy import copy
from typing import Dict, Iterable, Iterator, List, Optional, Set

import networkx as nx
import numpy as np
from networkx import NetworkXError
from networkx.classes.graph import _CachedPropertyResetterAdj

from pywhy_graphs.typing import Node, TsNode


class TsGraphNodeMixin:
    """A mixin class for dealing with nodes in time-series graph.

    Nodes in a ts-graph are characterized as a tuple of a variable and a time-index. This
    introduces also the notion of what is called a 'variable' inside a ts-graph. A variable
    is an entire time-series without the time-index.

    Notes
    -----
    A ts-graph's nodes are uniquely defined by the set of variables and the max-lag within
    the graph.
    """

    nodes: Iterator  # should be defined in parent class

    def _check_ts_node(self, node):
        if not isinstance(node, tuple) or len(node) != 2:
            raise ValueError(
                f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
                f"You passed in {node}."
            )
        if node[1] > 0:
            raise ValueError(f"All lag points should be 0, or less. You passed in {node}.")

        # Note: this uses the public max-lag, since the user should not be exposed to the
        # private max
        if node[1] < -self.max_lag:
            raise ValueError(f"Lag {node[1]} cannot be greater than set max_lag {self.max_lag}.")

    @property
    def variables(self) -> Set[Node]:
        """Set of variables in the time-series.

        Nodes in a time-series graph consist of variables X times.

        Returns
        -------
        variables : Set[Node]
            A set of variables.
        """
        node_vars = set()
        for node in self.nodes:
            node_vars.add(node[0])
        return node_vars

    def add_variable(self, variable: Node):
        # adding a node at t=0, should add to the rest of the graph
        self.add_node((variable, 0))

    def add_variables_from(self, variables: Iterable[Node]):
        for variable in variables:
            self.add_variable(variable)

    def nodes_at(self, t: int) -> Set:
        """Nodes at t=max_lag.

        Lag is a positive number, so a node at lag = 2,
        would be at time point "-2".
        """
        if t < 0:
            raise RuntimeError(f"Lag is a positive number. You passed {t}.")
        nodes = set()
        for node in self.nodes:
            if node[1] == -t:
                nodes.add(node)
        return nodes

    def add_node(self, node_name, **attr):
        """Add node in time."""
        self._check_ts_node(node_name)
        super().add_node(node_name, **attr)
        var_name, _ = node_name

        for t in range(self._max_lag + 1):
            super().add_node((var_name, -t), **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add nodes in time."""
        for node in nodes_for_adding:
            newdict = attr.copy()

            if isinstance(node[0], tuple):
                ndict = node[1]
                node = node[0]
                newdict = attr.copy()
                newdict.update(ndict)
            self.add_node(node, **newdict)

    def remove_node(self, node_name):
        """Remove node in time.

        Note
        ----
        Removing a node is not equivalent to adding a node. Addition/removal of
        nodes are provided as a convenience API, but time-series graphs operate
        by addition/removal of variables and modification of the max-lag.
        """
        self._check_ts_node(node_name)
        super().remove_node(node_name)

    def remove_nodes_from(self, ebunch):
        """Remove nodes in time."""
        for node in ebunch:
            self.remove_node(node)

    def remove_variable(self, variable_name):
        for lag in range(self.max_lag + 1):
            # we only remove nodes if they are within the set of noes
            node_name = (variable_name, -lag)
            if node_name in self.nodes:
                self.remove_node(node_name)

    def remove_variables_from(self, ebunch):
        for variable_name in ebunch:
            self.remove_variable(variable_name)

class TsGraphPropertyMixin:
    """A mixin class for time-series graph properties."""

    graph: Dict
    _adj: _CachedPropertyResetterAdj

    @property
    def max_lag(self) -> int:
        """The maximum time-index lag."""
        return self._max_lag

    @property
    def _max_lag(self) -> int:
        """Private property to query the maximum time-index lag stored in the graph.

        In stationary graphs, this is 2 times the maximum lag to enable proper
        d-separation querying.
        """
        return self.graph["max_lag"]

    def copy(self, as_view: bool = False):
        """Returns a copy of the graph.

        The copy method by default returns an independent shallow copy
        of the graph and attributes. That is, if an attribute is a
        container, that container is shared by the original an the copy.
        Use Python's `copy.deepcopy` for new containers.

        If `as_view` is True then a view is returned instead of a copy.

        Notes
        -----
        All copies reproduce the graph structure, but data attributes
        may be handled in different ways. There are four types of copies
        of a graph that people might want.

        Deepcopy -- A "deepcopy" copies the graph structure as well as
        all data attributes and any objects they might contain.
        The entire graph object is new so that changes in the copy
        do not affect the original object. (see Python's copy.deepcopy)

        Data Reference (Shallow) -- For a shallow copy the graph structure
        is copied but the edge, node and graph attribute dicts are
        references to those in the original graph. This saves
        time and memory but could cause confusion if you change an attribute
        in one graph and it changes the attribute in the other.
        NetworkX does not provide this level of shallow copy.

        Independent Shallow -- This copy creates new independent attribute
        dicts and then does a shallow copy of the attributes. That is, any
        attributes that are containers are shared between the new graph
        and the original. This is exactly what `dict.copy()` provides.
        You can obtain this style copy using:

            >>> G = nx.path_graph(5)
            >>> H = G.copy()
            >>> H = G.copy(as_view=False)
            >>> H = nx.Graph(G)
            >>> H = G.__class__(G)

        Fresh Data -- For fresh data, the graph structure is copied while
        new empty data attribute dicts are created. The resulting graph
        is independent of the original and it has no edge, node or graph
        attributes. Fresh copies are not enabled. Instead use:

            >>> H = G.__class__()
            >>> H.add_nodes_from(G)
            >>> H.add_edges_from(G.edges)

        View -- Inspired by dict-views, graph-views act like read-only
        versions of the original graph, providing a copy of the original
        structure without requiring any memory for copying the information.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data.

        Returns
        -------
        G : Graph
            A copy of the graph.

        See Also
        --------
        to_directed: return a directed copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.copy()

        """
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)

        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())  # type: ignore
        G.add_edges_from(  # type: ignore
            (u, v, datadict.copy())
            for u, nbrs in self._adj.items()
            for v, datadict in nbrs.items()
            if v[1] >= u[1]
        )

        return G


class TsGraphEdgePropertyMixin:
    """A mixin class for time-series graph edges.

    Adds the distinction between variables and nodes in a networkx-compliant
    graph. In addition, adds a ``max_lag`` parameter to keep track of how
    far back in terms of the time-index to keep track of.

    Also, adds distinction between contemporaneous and lagged edges, as well
    as contemporaneous and lagged neighbors.
    """

    graph: Dict
    edges: Iterator

    @property
    def contemporaneous_edges(self) -> List:
        """List of instantaneous (i.e. at same time point) edges."""
        edges = []
        for u_edge, v_edge in self.edges:
            if u_edge[1] == v_edge[1]:
                edges.append((u_edge, v_edge))
        return edges

    @property
    def lag_edges(self) -> List:
        """List of lagged edges."""
        edges = []
        for u_edge, v_edge in self.edges:
            if u_edge[1] < 0 and v_edge[1] > u_edge[1]:
                edges.append((u_edge, v_edge))
        return edges

    def lagged_neighbors(self, u):
        """Neighbors from t < u's current time index."""
        # DiGraph neighbors are defined, so if there is a notion of a
        # predecessor, we want all neighbors
        if hasattr(self, "predecessors"):
            nbrs = nx.all_neighbors(self, u)
        else:
            nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] < u[1]]

    def contemporaneous_neighbors(self, u):
        """Neighbors from the same time index as u."""
        # DiGraph neighbors are defined, so if there is a notion of a
        # predecessor, we want all neighbors
        if hasattr(self, "predecessors"):
            nbrs = nx.all_neighbors(self, u)
        else:
            nbrs = self.neighbors(u)
        return [nbr for nbr in nbrs if nbr[1] == u[1]]

    def set_max_lag(self, lag: int):
        """Set maximum-lag in time-series graph.

        By modifying the max-lag, certain nodes in time and edges will
        be either added, or removed.

        Parameters
        ----------
        lag : int
            The maximum lag (as a positive number).

        Returns
        -------
        self : time-series graph
            The modified time-series graph with new max-lag.
        """
        if lag <= 0:
            raise ValueError(
                f"Max lag must always be greater than 0, so passed in {lag} value is invalid."
            )
        max_lag = copy(self.max_lag)  # type: ignore
        self.graph["max_lag"] = lag

        # we need to add edges
        if lag > max_lag:
            # get all non-lag nodes
            non_lag_nodes = self.nodes_at(t=0)  # type: ignore

            # if we are dealing with a stationary graph, then we need
            # to add relevant edges to maintain stationary structure
            if self.stationary:
                # now get all neighbors that are in the past
                edge_list = []
                for node in non_lag_nodes:
                    edge_list.extend([(nbr, node) for nbr in self.lagged_neighbors(node)])

                # now add all homologous edges
                self.add_edges_from(edge_list)
            else:
                # just add relevant nodes
                for variable, _ in non_lag_nodes:
                    # all relevant nodes are now added based on new max-lag
                    self.add_node((variable, -lag))

        # here, we need to remove edges that are at higher lags
        elif max_lag > lag:
            for _lag in range(max_lag, lag, -1):
                # get all non-lag nodes
                nodes = self.nodes_at(t=-_lag)  # type: ignore
                self.remove_nodes_from(nodes)  # type: ignore
        return self


class TsGraphEdgeMixin:
    """A mixin class for time-series graph edges.

    Adds the distinction between variables and nodes in a networkx-compliant
    graph. In addition, adds a ``max_lag`` parameter to keep track of how
    far back in terms of the time-index to keep track of.

    Also, adds distinction between contemporaneous and lagged edges, as well
    as contemporaneous and lagged neighbors.
    """

    _auto_removal: Optional[str]
    graph: Dict
    _adj: _CachedPropertyResetterAdj
    edges: Iterator

    def add_edge(self, u_of_edge: TsNode, v_of_edge: TsNode, **attr):
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)
        u, u_lag = u_of_edge
        v, v_lag = v_of_edge
        if u_lag > 0 or v_lag > 0:
            raise RuntimeError(f"All lags should be negative or 0, not {u_lag} or {v_lag}.")

        # time-directionality should be checked if the graph is directed; for PAGs, we will
        # disable it
        if self.check_time_direction and v_lag < u_lag:
            raise RuntimeError(
                f'The lag of the "to node" {v_lag} should be greater than "from node" {u_lag}'
            )
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)

        if self.stationary:
            self.add_homologous_edges(u_of_edge, v_of_edge, **attr)
        else:
            super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_homologous_edges(self, u_of_edge: TsNode, v_of_edge: TsNode, direction="both", **attr):
        """Add homologous edges.

        Assumes the edge that we consider is ``(u_of_edge, v_of_edge)``, that is 'u' points to 'v'.

        Parameters
        ----------
        u_of_edge : TsNode
            The from node.
        v_of_edge : TsNode
            The to node. The absolute value of the time lag should be less than or equal to
            the from node's time lag.
        direction : str, optional
            Which direction to add homologous edges to, by default 'both', corresponding
            to making the edge stationary over all time.
        """
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)

        u, u_lag = u_of_edge
        v, v_lag = v_of_edge

        # take absolute value
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        if direction == "both":
            # re-center to 0, assuming v_lag is smaller, since it is the "to node"
            u_lag = u_lag - v_lag
            v_lag = 0

            # now add lagged edges up until max lag
            to_t = v_lag
            from_t = u_lag
            for _ in range(u_lag, self._max_lag + 1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t += 1
                from_t += 1
        elif direction == "forward":
            # decrease lag moving forward
            for _ in range(v_lag, -1, -1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t -= 1
                from_t -= 1
        elif direction == "backwards":
            for _ in range(u_lag, self._max_lag + 1):
                super().add_edge((u, -from_t), (v, -to_t), **attr)
                to_t += 1
                from_t += 1

    def remove_homologous_edges(self, u_of_edge: TsNode, v_of_edge: TsNode, direction="both"):
        """Remove homologous edges.

        Assumes the edge that we consider is ``(u_of_edge, v_of_edge)``, that is 'u' points to 'v'.

        Parameters
        ----------
        u_of_edge : TsNode
            The from node.
        v_of_edge : TsNode
            The to node. The absolute value of the time lag should be less than or equal to
            the from node's time lag.
        direction : str, optional
            Which direction to add homologous edges to, by default 'both', corresponding
            to making the edge stationary over all time.
        """
        self._check_ts_node(u_of_edge)
        self._check_ts_node(v_of_edge)

        u, u_lag = u_of_edge
        v, v_lag = v_of_edge

        # take absolute value
        u_lag = np.abs(u_lag)
        v_lag = np.abs(v_lag)

        if direction == "both":
            # re-center to 0, assuming v_lag is smaller, since it is the "to node"
            u_lag = u_lag - v_lag
            v_lag = 0

            # now add lagged edges up until max lag
            to_t = v_lag
            from_t = u_lag
            for _ in range(u_lag, self._max_lag + 1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t += 1
                from_t += 1
        elif direction == "forward":
            # decrease lag moving forward
            for _ in range(v_lag, -1, -1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t -= 1
                from_t -= 1
        elif direction == "backwards":
            for _ in range(u_lag, self._max_lag + 1):
                if self.has_edge((u, -from_t), (v, -to_t)):
                    super().remove_edge((u, -from_t), (v, -to_t))
                to_t += 1
                from_t += 1

    def add_edges_from(self, ebunch, **attr):
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesn't need edge_attr_dict_factory
            else:
                raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
            dd.update(attr)
            self.add_edge(u, v, **dd)

    def remove_edge(self, u_of_edge, v_of_edge, check_lag: bool = False):
        _, v_lag = v_of_edge
        if v_lag != 0 and check_lag:
            raise RuntimeError(f'The lag of the "to" node, {v_of_edge} should be 0.')

        if self.stationary:
            self.remove_homologous_edges(u_of_edge, v_of_edge)
        else:
            super().remove_edge(u_of_edge, v_of_edge)  # type: ignore

    def remove_edges_from(self, ebunch):
        for edge in ebunch:
            self.remove_edge(*edge)


class tsdict(dict):
    def __setitem__(self, key, val):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"All nodes in time series DAG must be a 2-tuple of the form (<node>, <lag>). "
                f"You passed in {key}."
            )
        if key[1] > 0:
            raise ValueError(f"All lag points should be 0, or less. You passed in {key}.")

        dict.__setitem__(self, key, val)


class BaseTimeSeriesGraph(
    TsGraphNodeMixin, TsGraphPropertyMixin, TsGraphEdgePropertyMixin, TsGraphEdgeMixin
):
    """A mixin class to imbue networkx graphs with time-series structure.

    This should not be used directly.

    Subclassing any time-series graph needs to define attributes and possibly override functions.

    Attributes
    ----------
    - stationary : bool
        Whether or not the graph should be assumed stationary. See Notes for details.
    - check_time_direction : bool
        Whether or not the graph should check for valid time-directionality when adding,
        or removing edges. For example, undirected graphs should set this to ``False`` to
        prevent errors due to the unordered edge structure. See Notes for details.

    Notes
    -----
    **How are time-series graphs different to networkx graphs?**

    A time-series graph is similar to a normal NetworkX graph, except now each node is
    characterized by a tuple of the variable (i.e. 'A', 'B', 'X'), and the time-index lag
    (i.e. 0, -1, -4). A node would be ``('A', 0)``, indicating the variable 'A' at time
    lag 0. Therefore, any time-series graph's nodes are completely defined if
    given a set of variables and a max-lag. For example, if we have the following variable
    time-series 'x', 'y', 'z', and a max-lag of 2, then including time-lag of 0, there are
    three time points for every variable, resulting in nine unique nodes in the graph.

    Whenever a node is added with a variable that is not present in the graph, then all
    time-indices of that variable will be added too. For example:

    > # Adding variable 'x' at time-lag 0 where 'x' is not in the variables of the graph
    > G.add_node(('x', 0))
    > G.has_node(('x', G.max_lag))
    > True

    **Stationary vs nonstationary time-series graphs**

    A time-series graph is by default not stationary. A stationary time-series graph is
    where all edges repeat over time. So looking at all edges with nodes at time-lag t=0
    is sufficient to determine the edge structure over the entire max-lag. For example,
    if max-lag is 3 and there are edges ``[(('x', -1), ('x', 0)), (('x', -2), ('x', 0))]``,
    then there is also the edges
    ``[(('x', -2), ('x', -1)), (('x', -3), ('x', -2)), (('x', -3), ('x', -1))]`` when
    a graph is assumed to be stationary. When edges are _added/removed_ in a stationary graph,
    other edges are _automatically added and removed_ to keep the stationary edge structure.
    If you do not want this feature, then your time-series graph must have ``stationary`` property
    set to ``False``. Even if the graph is considered nonstationary, a user may still manually
    add/remove homologous edges. They may even specify the direction that this occurs.
    """

    # whether or not the graph should be assumed to be stationary
    stationary: bool = False

    # whether to check for valid time-directionality in edges
    check_time_direction: bool = False

    # overloaded factory dictionary types to hold time-series nodes
    node_dict_factory = tsdict
    node_attr_dict_factory = tsdict
    adjlist_outer_dict_factory = tsdict
    adjlist_inner_dict_factory = tsdict
