import networkx as nx
import pytest

import pywhy_graphs.networkx as pywhy_nx
from pywhy_graphs.algorithms.cpdag import cpdag_to_pdag, pdag_to_dag


class TestPDAGtoDAG:
    def test_pdag_to_dag_errors(self):
        G = nx.DiGraph()
        G.add_edges_from([("X", "Y"), ("Z", "X")])
        G.add_edge("A", "Z")
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "Z")])
        G = pywhy_nx.MixedEdgeGraph(graphs=[G], edge_types=["directed"], name="IV Graph")
        G.add_edge_type(nx.DiGraph(), "circle")
        G.add_edge("Z", "A", edge_type="circle")
        G.add_edge("A", "B", edge_type="circle")
        G.add_edge("B", "A", edge_type="circle")
        G.add_edge("B", "Z", edge_type="circle")
        G = cpdag_to_pdag(G)
        with pytest.raises(
            ValueError, match="Only directed and undirected edges are allowed in a CPDAG"
        ):
            pdag_to_dag(G)

    def test_pdag_to_dag_1(self):
        G = nx.DiGraph()
        G.add_edges_from([("X", "Y"), ("Z", "X")])
        G.add_edge("A", "Z")
        G.add_edges_from([("A", "B"), ("B", "A"), ("B", "Z")])
        G = pywhy_nx.MixedEdgeGraph(graphs=[G], edge_types=["directed"], name="IV Graph")
        G.add_edge_type(nx.DiGraph(), "circle")
        G.add_edge("Z", "A", edge_type="circle")
        G.add_edge("A", "B", edge_type="circle")
        G.add_edge("B", "A", edge_type="circle")
        G.add_edge("B", "Z", edge_type="circle")
        G = cpdag_to_pdag(G)
        G = pdag_to_dag(G)
        assert G.edges == {("X", "Y"), ("Z", "X"), ("A", "Z"), ("A", "B"), ("B", "Z")}
