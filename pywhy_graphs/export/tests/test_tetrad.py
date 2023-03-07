from pathlib import Path

import pywhy_graphs
from pywhy_graphs.export import graph_to_tetrad, tetrad_to_graph
from pywhy_graphs.testing import assert_mixed_edge_graphs_isomorphic

root = Path(pywhy_graphs.__file__).parents[1]
filename = root / "pywhy_graphs" / "export" / "tests" / "test_pag_tetrad.txt"


def test_to_tetrad(tmp_path):
    G = tetrad_to_graph(filename, graph_type="pag")

    tmp_fname = Path(tmp_path) / "test.txt"
    graph_to_tetrad(G, tmp_fname)

    roundtrip_G = tetrad_to_graph(filename, graph_type="pag")

    assert_mixed_edge_graphs_isomorphic(G, roundtrip_G)
