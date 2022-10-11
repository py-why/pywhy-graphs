import dynetx
import networkx as nx
import pytest

from pywhy_graphs.classes.tsbase import TimeSeriesMixedEdgeGraph


class TestTimeSeriesMixedEdgeGraph:
    def setup(self):
        self.G = TimeSeriesMixedEdgeGraph()

    def test_nodes(self):
        g = TimeSeriesMixedEdgeGraph
        g.add_interaction(0, 1, t=5)
        nds = len(g.nodes())
        assert nds == 2

        g.add_interaction(1, 2, t=6)
        nds = len(g.nodes())
        assert nds == 3

        nds = len(g.nodes(t=6))
        assert nds == 2

        nds = len(g.nodes(t=9))
        assert nds == 0

        assert g.has_node(0)
        assert g.has_node(0, 5)
        assert not g.has_node(0, 6)
        assert not g.has_node(0, 0)


def test_timeseriesmixededgegraph():
    G = TimeSeriesMixedEdgeGraph()
