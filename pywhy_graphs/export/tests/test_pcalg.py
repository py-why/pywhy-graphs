import numpy as np
from numpy.testing import assert_array_equal

from pywhy_graphs.config import PCAlgCPDAGEndpoint, PCAlgPAGEndpoint
from pywhy_graphs.export import graph_to_pcalg, pcalg_to_graph


def test_pag_pcalg_to_graph():
    pcalg_arr = np.array(
        [
            [0, PCAlgPAGEndpoint.ARROW.value, PCAlgPAGEndpoint.CIRCLE.value],
            [PCAlgPAGEndpoint.ARROW.value, 0, PCAlgPAGEndpoint.CIRCLE.value],
            [PCAlgPAGEndpoint.ARROW.value, PCAlgPAGEndpoint.TAIL.value, 0],
        ]
    )

    G = pcalg_to_graph(pcalg_arr, arr_idx=[0, 1, 2], amat_type="pag")

    pcalg_arr_rt = graph_to_pcalg(G)
    assert_array_equal(pcalg_arr, pcalg_arr_rt)


def test_cpdag_pcalg_to_graph():
    pcalg_arr = np.array(
        [
            [0, PCAlgCPDAGEndpoint.ARROW.value, PCAlgCPDAGEndpoint.NULL.value],
            [PCAlgCPDAGEndpoint.ARROW.value, 0, PCAlgCPDAGEndpoint.ARROW.value],
            [PCAlgCPDAGEndpoint.ARROW.value, PCAlgCPDAGEndpoint.NULL.value, 0],
        ]
    )

    G = pcalg_to_graph(pcalg_arr, arr_idx=[0, 1, 2], amat_type="cpdag")
    print(G.edges())
    pcalg_arr_rt = graph_to_pcalg(G)
    assert_array_equal(pcalg_arr, pcalg_arr_rt)
