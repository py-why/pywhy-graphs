import numpy as np
from numpy.testing import assert_array_equal

from pywhy_graphs.config import PCAlgPAGEndpoint
from pywhy_graphs.export import graph_to_pcalg, pcalg_to_graph


def test_pcalg_to_graph():
    pcalg_arr = np.array(
        [
            [0, PCAlgPAGEndpoint.ARROW.value, PCAlgPAGEndpoint.CIRCLE.value],
            [PCAlgPAGEndpoint.ARROW.value, 0, PCAlgPAGEndpoint.CIRCLE.value],
            [PCAlgPAGEndpoint.ARROW.value, PCAlgPAGEndpoint.TAIL.value, 0],
        ]
    )

    G = pcalg_to_graph(pcalg_arr, arr_idx=[0, 1, 2], amat_type="pag")

    pcalg_arr_rt = graph_to_pcalg(G)
    print(G)
    from pprint import pprint

    pprint(G.edges())
    assert_array_equal(pcalg_arr, pcalg_arr_rt)
