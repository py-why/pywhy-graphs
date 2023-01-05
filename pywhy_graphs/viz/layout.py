import numpy as np
from networkx.drawing.layout import _process_params, rescale_layout


def timeseries_layout(G, variable_order=None, scale=5, center=None, aspect_ratio=4 / 3):
    """Position nodes in a time-series layout from left to right with lags as columns.

    Parameters
    ----------
    G : TimeSeriesGraph
        A timeseries graph.
    variable_order : list, optional
        List of variables in ``G`` to order from top to bottom, by default None, which
        would be a random order.
    scale : int, optional
        Scale factor for positions, by default 5.
    center : ArrayLike, optional
        The 2D array of the center, by default None, which will plot around the point (0, 0).
    aspect_ratio : float, optional
        The ratio of the width to the height of the layout, by default 4/3.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Notes
    -----
    The time-series layout lays out nodes from historical lags to the present (i.e. max-lag to t=0)
    in left to right fashion. It also keeps each row set as a specific time-series variable.
    """
    G, center = _process_params(G, center=center, dim=2)
    if len(G.nodes) == 0:
        return {}

    height = 10
    width = aspect_ratio * height

    # spacing between rows and columns
    # offset = (0., 0.)
    offset = (width / 2, height / 2)

    if variable_order is None:
        variable_order = G.variables

    # get the max-lag, which will dictate the x-spacing
    max_lag = G.max_lag

    # define x and y spacing
    xs = np.linspace(0, width, max_lag + 1)
    ys = np.linspace(0, height, len(variable_order))

    # create positions for every node
    pos = np.zeros((G.number_of_nodes(), 2))
    nodes = []
    irow = 0
    for idy, variable in enumerate(variable_order):
        for idx, lag in enumerate(range(max_lag + 1)):
            x = xs[-(idx + 1)]
            y = ys[idy]
            pos[irow, :] = np.array([x, y]) - offset
            nodes.append((variable, -lag))
            irow += 1

    # post-process the layout
    pos = np.array(pos)
    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(nodes, pos))
    return pos
