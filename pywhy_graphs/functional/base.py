from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

import pywhy_graphs as pgraphs
from pywhy_graphs import AugmentedGraph
from pywhy_graphs.typing import Node


def add_parent_function(G: nx.DiGraph, node: Node, func: Callable) -> nx.DiGraph:
    """Add parent function for a node into the graph.

    Parameters
    ----------
    G : nx.DiGraph
        A DAG.
    node : Node
        The node in which to add a functional relationship from its parents.
    func : Callable
        A callable that takes in the values of the parents and returns a
        value.

    Returns
    -------
    G : nx.DiGraph
        The DAG with the parent function added to ``G.node[node]["parent_function"]``.
    """
    # get the parents of the current node
    parents = list(G.predecessors(node))

    # check that the function has the correct number of keyword arguments
    _check_input_func(func, parents)

    G.nodes[node]["parent_function"] = func
    G.graph["functional"] = True
    return G


def add_noise_function(
    G: nx.DiGraph, node: Node, distr_func: Callable, func: Callable = None
) -> nx.DiGraph:
    """Add function and distribution for a node's exogenous variable into the graph.

    Parameters
    ----------
    G : nx.DiGraph
        A DAG.
    node : Node
        The node in which to add a functional relationship with respect to
        a random exogenous variable.
    distr_func : Callable
        A callable that generates the random exogenous variable. For example, this
        can be an instance of the standard normal distribution, or the uniform
        distribution.
    func : Callable
        A callable that takes in the values of the exogenous variable and returns
        a value. If None, then the identity function is used.

    Returns
    -------
    G : nx.DiGraph
        The DAG with the exogenous function added to ``G.node[node]["exogenous_function"]``.
    """
    if func is None:
        func = lambda x: x
    else:
        # check that the function has the correct number of keyword arguments
        _check_input_func(func)
    if distr_func.__code__.co_argcount + +func.__code__.co_kwonlyargcount > 0:
        raise ValueError(
            f"Function {distr_func} for the exogenous variable distribution takes in "
            f"too many arguments. It should take in 0 arguments."
        )

    G.nodes[node]["exogenous_function"] = func
    G.nodes[node]["exogenous_distribution"] = distr_func
    G.graph["functional"] = True
    return G


def add_soft_intervention_function(
    G: AugmentedGraph, node: Node, f_node: Node, func: Callable
) -> nx.DiGraph:
    """Add soft intervention function for a node into the graph.

    Parameters
    ----------
    G : nx.DiGraph
        A DAG.
    node : Node
        The node in which to add a functional relationship from its parents.
    f_node : Node
        The F-node which represents this specific intervention.
    func : Callable
        A callable that takes in the values of the parents of ``node`` and returns a
        value.

    Returns
    -------
    G : nx.DiGraph
        The DAG with the parent function added to
        ``G.node[node]["intervention_functions"][f_node]``.
    """
    if node not in G.children(f_node):
        raise RuntimeError(f"Node {node} is not a child of F-node {f_node}.")

    # get the observed variable parents of the current node
    parents = set(G.predecessors(node)).difference(set(G.augmented_nodes))

    # check that the function has the correct number of keyword arguments
    _check_input_func(func, parents=parents)

    G.nodes[node]["intervention_functions"][f_node] = func
    G.graph["functional"] = True
    G.graph["interventional"] = True
    return G


def add_domain_shift_function(
    G: AugmentedGraph, node: Node, s_node: Node, func: Callable = None, distr_func: Callable = None
):
    """Add domain shift function for a node into the graph assuming invariant graph structure.

    A domain shift can either change the functional relationship of the node, or change the
    distribution of the exogenous variables. This is known as a mechanism change in
    selection diagrams. This function assumes that the graph structure is invariant
    before and after the domain shift.

    Parameters
    ----------
    G : AugmentedGraph
        The augmented graph with S-nodes.
    node : Node
        The node ``s_node`` is pointing to.
    s_node : Node
        The S-node indicating a change in distribution between the two domains for the
        nodes it is pointing to.
    func : Callable, optional
        New function for ``node`` for the domain, by default None
    distr_func : Callable, optional
        _description_, by default None
    """
    if node not in G.children(s_node):
        raise RuntimeError(f"Node {node} is not a child of S-node {s_node}.")
    if distr_func is None and func is None:
        raise RuntimeError("Either func or distr_func must be specified.")

    # get the observed variable parents of the current node
    parents = set(G.predecessors(node)).difference(set(G.augmented_nodes))

    # get the domain ID of the S-node in sorted order (domain_i, domain_j),
    # where domain_i < domain_j
    domain_ids = sorted(G.nodes[s_node]["domain_ids"])
    src, target = domain_ids
    reference_src = min(G.domain_ids)

    # check that the function has the correct number of keyword arguments
    if func is not None:
        _check_input_func(func, parents)
    else:
        # use the existing function
        func = G.nodes[node].get("parent_function", None)
        if func is None:
            raise RuntimeError(
                f"Node {node} does not have a parent function and func is not specified."
            )
    if distr_func is None:
        distr_func = G.nodes[node].get("exogenous_distribution", None)
        if distr_func is None:
            raise RuntimeError(
                f"Node {node} does not have an exogenous distribution and distr_func "
                f"is not specified."
            )

    # check the existing functions are added for the src
    if src != reference_src and src not in G.nodes[node]["domain_parent_functions"]:
        raise RuntimeError(f"Node {node} does not have a parent function for domain {src} yet.")
    if src != reference_src and src not in G.nodes[node]["domain_exogenous_distributions"]:
        raise RuntimeError(
            f"Node {node} does not have an exogenous distribution for domain {src} yet."
        )

    G.nodes[node]["domain_parent_functions"][target] = func
    G.nodes[node]["domain_exogenous_distributions"][target] = distr_func
    G.graph["functional"] = True
    G.graph["multidomain"] = True
    return G


def sample_from_graph(
    G: nx.DiGraph,
    n_samples: int = 1000,
    n_jobs: Optional[int] = None,
    random_state=None,
    **sample_kwargs,
):
    """Sample a dataset from a linear Gaussian graph.

    Assumes the graph only consists of directed edges. It is on the roadmap to
    implement support for bidirected edges.

    Parameters
    ----------
    G : Graph
        A linear DAG from which to sample. Must have been set up with
        :func:`pywhy_graphs.functional.make_graph_linear_gaussian`.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    n_jobs : Optional[int], optional
        Number of jobs to run in parallel, by default None.
    random_state : int, optional
        Random seed, by default None.
    **sample_kwargs
        Keyword arguments to pass to the sampling function.

    Returns
    -------
    data : pd.DataFrame of shape (n_samples, n_nodes)
        A pandas DataFrame with the iid samples.
    """
    import pandas as pd

    rng = np.random.default_rng(random_state)
    if hasattr(G, "get_graphs"):
        directed_G = G.get_graphs("directed")
    else:
        directed_G = G

    # check input
    _check_input_graph(directed_G)

    # first off, always convert said graph into an AugmentedGraph
    if isinstance(G, nx.DiGraph):
        G = pgraphs.AugmentedGraph(**G.graph)
        for node in directed_G.nodes:
            G.add_node(node, **directed_G.nodes[node])
        for edge in directed_G.edges:
            G.add_edge(*edge, **directed_G.edges[edge])

    # Create list of topologically sorted nodes
    top_sort_idx = list(nx.topological_sort(directed_G))

    if hasattr(G, "augmented_nodes"):
        top_sort_idx = [node for node in top_sort_idx if node not in G.augmented_nodes]
        ignored_nodes = G.augmented_nodes
    else:
        ignored_nodes = None

    # Sample from graph
    if n_jobs == 1:
        data = []
        for _ in range(n_samples):
            node_samples = _sample_from_graph(
                G, top_sort_idx, rng=rng, ignored_nodes=ignored_nodes, **sample_kwargs
            )
            data.append(node_samples)
        data = pd.DataFrame.from_records(data)
    else:
        out = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_sample_from_graph)(
                G, top_sort_idx, rng=rng, ignored_nodes=ignored_nodes, **sample_kwargs
            )
            for _ in range(n_samples)
        )
        data = pd.DataFrame.from_records(out)

    return data


def _sample_from_graph(
    G: nx.DiGraph,
    top_sort_idx,
    rng,
    ignored_nodes=None,
) -> Dict:
    """Private function to sample a single iid sample from a graph for all nodes.

    Parameters
    ----------
    G : nx.DiGraph
        The DAG.
    top_sort_idx : List of node
        The topologically sorted nodes.
    ignored_nodes : _type_, optional
        Nodes that should not be sample, by default None.

    Returns
    -------
    nodes_sample : dict
        The sample per node.
    """
    if ignored_nodes is None:
        ignored_nodes = set()

    nodes_sample: Dict = dict()

    for node in top_sort_idx:
        # get all parents
        parents = list(set(G.predecessors(node)).difference(ignored_nodes))

        if G.graph["functional"] == "discrete":
            # for discrete graphs, use noise ratio to determine whether we sample
            # from noise, or from the CPT
            noise_ratio = G.nodes[node]["noise_ratio"]
            p = [noise_ratio, 1 - noise_ratio]
            if rng.choice(["exogenous", "cpt"], p=p) == "exogenous":
                exo_val = G.nodes[node]["exogenous_distribution"]()
            else:
                exo_val = 0.0

            exo_contrib_node = G.nodes[node]["exogenous_function"](exo_val)
        else:
            # XXX: need to verify that this is reproducible given a RNG passed to the original func
            # sample exogenous variable, which has no parameters
            exo_val = G.nodes[node]["exogenous_distribution"]()
            exo_contrib_node = G.nodes[node]["exogenous_function"](exo_val)

        # sample parents if they exist
        parents_contrib_node = 0.0
        if parents:
            sorted_idx = np.argsort(parents)
            parent_vals = [nodes_sample[parents[idx]] for idx in sorted_idx]

            parents_contrib_node = G.nodes[node]["parent_function"](*parent_vals)

        # set the node attribute "functions" to hold the weight and function wrt each parent
        node_sample = parents_contrib_node + exo_contrib_node
        nodes_sample[node] = node_sample
    return nodes_sample


def _check_input_func(func: Callable, parents=None):
    # check exogenous function
    if parents is None:
        if func.__code__.co_argcount + func.__code__.co_kwonlyargcount != 1:
            raise ValueError(
                f"Function {func} should have only 1 argument that accepts a random, "
                f"variable from the exogenous variable's distribution, "
                f"but has {func.__code__.co_argcount + func.__code__.co_kwonlyargcount} "
                f"arguments."
            )
    else:
        if func.__code__.co_kwonlyargcount != len(parents):
            raise ValueError(f"Function {func} should have {len(parents)} keyword-only arguments, ")


def _check_input_graph(G: nx.DiGraph):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph must be a DAG.")
    if not G.graph.get("functional", True):
        raise ValueError(
            "The input graph must be a functional graph. Please initialize "
            "the graph with functions."
        )
    for node in G.nodes:
        if G.nodes[node].get("exogenous_function", None) is None:
            raise ValueError(f"Node {node} does not have an exogenous function.")
        if G.nodes[node].get("exogenous_distribution", None) is None:
            raise ValueError(f"Node {node} does not have an exogenous variable distribution.")
        if (
            G.nodes[node].get("parent_function", None) is None
            and len(list(G.predecessors(node))) > 0
        ):
            raise ValueError(f"Node {node} does not have a parent function, but it has parents.")
        if (
            G.nodes[node].get("parent_function", None) is not None
            and len(list(G.predecessors(node))) == 0
        ):
            raise ValueError(f"Node {node} has a parent function, but it has no parents.")
