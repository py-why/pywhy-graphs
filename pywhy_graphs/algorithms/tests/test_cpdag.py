import networkx as nx
import numpy as np
import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.algorithms import all_vstructures
from pywhy_graphs.algorithms.cpdag import (
    EDGELABELS,
    dag_to_cpdag,
    label_edges,
    order_edges,
    pdag_to_cpdag,
    pdag_to_dag,
)
from pywhy_graphs.testing import assert_mixed_edge_graphs_isomorphic

seed = 12345
rng = np.random.default_rng(seed)


class TestOrderEdges:
    def test_order_edges_errors(self):
        G = nx.DiGraph()

        # 1 -> 2 -> 4 -> 5
        # 1 -> 3 -> 4
        # so topological sort is: (1, 2, 3, 4, 5)
        G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
        # now test when there is a cycle
        G.add_edge(5, 1)
        with pytest.raises(ValueError, match="G must be a directed acyclic graph"):
            order_edges(G)

    def test_order_edges(self):
        # Example usage:
        G = nx.DiGraph()

        # 1 -> 2 -> 4 -> 5
        # 1 -> 3 -> 4
        G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
        G = order_edges(G)

        expected_order = [
            (1, 2, {"order": 4}),
            (1, 3, {"order": 3}),
            (2, 4, {"order": 1}),
            (3, 4, {"order": 2}),
            (4, 5, {"order": 0}),
        ]
        assert set(G.edges.data(data="order")) == set(
            [(src, target, order["order"]) for src, target, order in expected_order]
        )

        # Add a string as a node
        # 5 -> 3 -> 1 -> 2 -> 'a'; 1 -> 'b'
        G = nx.DiGraph()
        G.add_edges_from([(5, 3), (3, 1), (1, 2), (2, "a"), (1, "b")])
        G = order_edges(G)

        expected_order = [
            (5, 3, {"order": 4}),
            (3, 1, {"order": 3}),
            (1, 2, {"order": 2}),
            (1, "b", {"order": 1}),
            (2, "a", {"order": 0}),
        ]
        assert set(G.edges.data(data="order")) == set(
            [(src, target, order["order"]) for src, target, order in expected_order]
        )

    def test_order_edges_ex1(self):
        G = nx.DiGraph()

        # 1 -> 3; 1 -> 4; 1 -> 5;
        # 2 -> 3; 2 -> 4; 2 -> 5;
        # 3 -> 4; 3 -> 5;
        # 4 -> 5;
        G.add_edges_from([(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
        G = order_edges(G)
        expected_order = [
            (1, 3, {"order": 7}),
            (1, 4, {"order": 4}),
            (1, 5, {"order": 0}),
            (2, 3, {"order": 8}),
            (2, 4, {"order": 5}),
            (2, 5, {"order": 1}),
            (3, 4, {"order": 6}),
            (3, 5, {"order": 2}),
            (4, 5, {"order": 3}),
        ]
        assert set(G.edges.data(data="order")) == set(
            [(src, target, order["order"]) for src, target, order in expected_order]
        )


class TestLabelEdges:
    def test_label_edges_raises_error_for_non_dag(self):
        # Test that label_edges raises a ValueError for a non-DAG
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])  # A cyclic graph
        with pytest.raises(ValueError, match="G must be a directed acyclic graph"):
            label_edges(G)

    def test_label_edges_raises_error_for_unordered_edges(self):
        # Test that label_edges raises a ValueError for unordered edges
        G = nx.DiGraph([(1, 2), (2, 3)])
        with pytest.raises(
            ValueError, match="G must have all edges ordered via the `order` attribute"
        ):
            label_edges(G)

    def test_label_edges_all_compelled(self):
        # Create an example DAG for testing
        G = nx.DiGraph()

        # 1 -> 3; 3 -> 4; 3 -> 5
        # 2 -> 3;
        # 4 -> 5
        G.add_edges_from([(1, 3), (2, 3), (3, 4), (3, 5), (4, 5)])
        nx.set_edge_attributes(G, None, "order")
        G = order_edges(G)
        labeled_graph = label_edges(G)

        expected_labels = {
            (1, 3): EDGELABELS.COMPELLED,
            (2, 3): EDGELABELS.COMPELLED,
            (3, 4): EDGELABELS.COMPELLED,
            (3, 5): EDGELABELS.COMPELLED,
            (4, 5): EDGELABELS.REVERSIBLE,
        }
        for edge, expected_label in expected_labels.items():
            assert labeled_graph[edge[0]][edge[1]]["label"] == expected_label, (
                f"Edge {edge} has label {labeled_graph[edge[0]][edge[1]]['label']}, "
                f"but expected {expected_label}"
            )


class TestPDAGtoDAG:
    def test_pdag_to_dag_errors(self):
        G = nx.DiGraph()
        G.add_edge("A", "Z")
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "Z"), ("X", "Y"), ("Z", "X")])

        # add non-CPDAG supported edges
        G = pywhy_nx.MixedEdgeGraph(graphs=[G], edge_types=["directed"])
        G.add_edge_type(nx.DiGraph(), "circle")
        G.add_edge("Z", "A", edge_type="circle")
        G.add_edge("A", "B", edge_type="circle")
        G.add_edge("B", "A", edge_type="circle")
        G.add_edge("B", "Z", edge_type="circle")
        with pytest.raises(
            ValueError, match="Only directed and undirected edges are allowed in a CPDAG"
        ):
            pdag_to_dag(G)

    def test_pdag_to_dag_inconsistent(self):
        # 1 -- 3; 1 -> 4;
        # 2 -> 3;
        # 4 -> 3
        # Note: this PDAG is inconsistent because it would create a v-structure, or a cycle
        # by orienting the undirected edge 1 -- 3
        pdag = pywhy_nx.MixedEdgeGraph(
            graphs=[nx.DiGraph(), nx.Graph()], edge_types=["directed", "undirected"]
        )
        pdag.add_edge(1, 3, edge_type="undirected")
        pdag.add_edges_from([(1, 4), (2, 3), (4, 3)], edge_type="directed")
        with pytest.raises(ValueError, match="No consistent extension found"):
            pdag_to_dag(pdag)

    def test_pdag_to_dag_already_dag(self):
        # 1 -> 2; 1 -> 3
        # 2 -> 3
        # 4 -> 3
        pdag = pywhy_nx.MixedEdgeGraph(
            graphs=[nx.DiGraph(), nx.Graph()], edge_types=["directed", "undirected"]
        )
        pdag.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 3)], edge_type="directed")
        G = pdag_to_dag(pdag)
        assert nx.is_isomorphic(G, pdag.get_graphs("directed"))

    def test_pdag_to_dag_0(self):
        # 1 -- 3;
        # 2 -> 3; 2 -> 4
        pdag = pywhy_nx.MixedEdgeGraph(
            graphs=[nx.DiGraph(), nx.Graph()], edge_types=["directed", "undirected"]
        )

        pdag.add_edge(1, 3, edge_type="undirected")
        pdag.add_edges_from([(2, 3), (2, 4)], edge_type="directed")

        G = pdag_to_dag(pdag)

        # add a directed edge from 3 to 1
        pdag.remove_edge(1, 3, edge_type="undirected")
        pdag.add_edge(3, 1, edge_type="directed")

        assert nx.is_isomorphic(G, pdag.get_graphs("directed"))

    def test_pdag_to_dag_1(self):
        # 1 -- 3;
        # 2 -> 1; 2 -> 4
        pdag = pywhy_nx.MixedEdgeGraph(
            graphs=[nx.DiGraph(), nx.Graph()], edge_types=["directed", "undirected"]
        )

        pdag.add_edge(1, 3, edge_type="undirected")
        pdag.add_edges_from([(2, 1), (2, 4)], edge_type="directed")

        G = pdag_to_dag(pdag)
        pdag.remove_edge(1, 3, edge_type="undirected")
        pdag.add_edge(1, 3, edge_type="directed")

        assert nx.is_isomorphic(G, pdag.get_graphs("directed"))

    def test_pdag_to_cpdag(self):
        # construct a random DAG
        n = 10
        p = 0.4
        random_graph = nx.fast_gnp_random_graph(n, p, directed=True, seed=seed)
        dag = nx.DiGraph([(u, v) for (u, v) in random_graph.edges() if u < v])

        pdag = pywhy_nx.MixedEdgeGraph(
            graphs=[dag.copy(), nx.Graph()], edge_types=["directed", "undirected"]
        )

        # now we construct the set of undirected edges that to not belong
        # to any unshielded collider (i.e. v-structure)
        vstructs = all_vstructures(dag, as_edges=True)

        # we apply a random orientation for a subset of the undirected edges
        for edge in dag.edges:
            if edge not in vstructs:
                if rng.binomial(1, 0.3):
                    pdag.remove_edge(*edge)
                    pdag.add_edge(*edge, edge_type="undirected")

        # now, we can convert the DAG to CPDAG and also convert the PDAG to a CPDAG
        # they should be equivalent
        cpdag = dag_to_cpdag(dag)
        cpdag_from_pdag = pdag_to_cpdag(pdag)

        assert_mixed_edge_graphs_isomorphic(cpdag, cpdag_from_pdag)
