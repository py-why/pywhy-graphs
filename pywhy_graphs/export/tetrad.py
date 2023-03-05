from typing import Dict

import pywhy_graphs
from pywhy_graphs.config import TetradEndpoint


def tetrad_to_graph(filename: str, graph_type):
    """Convert a tetrad stored graph from a text file to causal graph in pywhy.

    Parameters
    ----------
    filename : str
        The file at which the tetrad file is stored.
    graph_type : str
        Type of pywhy causal graph to construct.
        One of ('pag', 'cpdag', 'admg', 'dag').

    Returns
    -------
    G : instance of causal graph
        The causal graph.

    Notes
    -----
    See tetrad documentation: https://cmu-phil.github.io/tetrad/manual/.

    Warning: There is no guarantee or check that your file adheres to the
    tetrad format.
    """
    # instantiate the type of causal graph
    if graph_type == "dag":
        G = pywhy_graphs.ADMG()
    elif graph_type == "admg":
        G = pywhy_graphs.ADMG()
    elif graph_type == "cpdag":
        G = pywhy_graphs.CPDAG()  # type: ignore
    elif graph_type == "pag":
        G = pywhy_graphs.PAG()
    elif graph_type in [pywhy_graphs.ADMG, pywhy_graphs.CPDAG, pywhy_graphs.PAG]:
        G = graph_type()
    else:
        raise RuntimeError(
            f"The graph type {graph_type} is unrecognized. Please use one of "
            f"'dag', 'admg', 'cpdag', 'pag'."
        )

    with open(filename, "r") as file:
        next_nodes_line = False
        for line in file.readlines():
            line = line.strip()
            words = line.split()

            # add nodes to the graph
            if len(words) > 1 and words[1] == "Nodes:":
                next_nodes_line = True
            elif len(line) > 0 and next_nodes_line:
                next_nodes_line = False
                nodes = line.split(";")
                for node in nodes:
                    G.add_node(node)

            # add edges
            elif len(words) > 0 and words[0][-1] == ".":
                next_nodes_line = False
                node1 = words[1]
                node2 = words[3]
                end1 = words[2][0]
                end2 = words[2][-1]
                if end1 == "<":
                    end1 = ">"

                if end1 == TetradEndpoint.ARROW.value:
                    if end2 == TetradEndpoint.ARROW.value:
                        G.add_edge(node1, node2, G.bidirected_edge_name)
                    elif end2 == TetradEndpoint.TAIL.value:
                        G.add_edge(node1, node2, G.directed_edge_name)
                    elif end2 == TetradEndpoint.CIRCLE.value:
                        G.add_edge(node1, node2, G.circle_edge_name)
                        G.add_edge(node2, node1, G.directed_edge_name)
                elif end1 == TetradEndpoint.TAIL.value:
                    if end2 == TetradEndpoint.ARROW.value:
                        G.add_edge(node2, node1, G.directed_edge_name)
                    elif end2 == TetradEndpoint.TAIL.value:
                        G.add_edge(node2, node1, G.undirected_edge_name)
                    elif end2 == TetradEndpoint.CIRCLE.value:
                        G.add_edge(node1, node2, G.circle_edge_name)
                elif end1 == TetradEndpoint.CIRCLE.value:
                    if end2 == TetradEndpoint.ARROW.value:
                        G.add_edge(node1, node2, G.directed_edge_name)
                        G.add_edge(node2, node1, G.circle_edge_name)
                    elif end2 == TetradEndpoint.TAIL.value:
                        G.add_edge(node2, node1, G.circle_edge_name)
                    elif end2 == TetradEndpoint.CIRCLE.value:
                        G.add_edge(node1, node2, G.circle_edge_name)
                        G.add_edge(node2, node1, G.circle_edge_name)
    return G


def graph_to_tetrad(G, filename: str):
    """Convert a pywhy causal graph to a tetrad text file.

    Parameters
    ----------
    G : instance of causal graph
        Causal graph.
    filename : str
        Output text file to write tetrad formatted graph.
    """
    tetrad_txt = "Graph Nodes:\n"

    graph_edge_dict: Dict = dict()
    for idx, node in enumerate(G.nodes):
        if idx == 0:
            tetrad_txt += f"{node}"
        else:
            tetrad_txt += f";{node}"

        # process all edge
        if node not in graph_edge_dict:
            graph_edge_dict[node] = dict()

        # get all neighbors
        nbrs = G.neighbors(node)

        # for each neighbor, get the type of edges
        node_nbr_str = ""
        for nbr in nbrs:
            if nbr not in graph_edge_dict:
                graph_edge_dict[nbr] = dict()
            if nbr in graph_edge_dict[node] or node in graph_edge_dict[nbr]:
                continue

            # process edge types among all possible nodes
            if G.has_edge(node, nbr, G.directed_edge_name):
                if not G.has_edge(nbr, node):
                    node_nbr_str = "-->"
                elif G.has_edge(nbr, node, G.circle_edge_name):
                    node_nbr_str = "o->"
            elif G.has_edge(node, nbr, G.bidirected_edge_name):
                node_nbr_str = "<->"
            elif G.has_edge(node, nbr, G.undirected_edge_name):
                node_nbr_str = "---"

            graph_edge_dict[node][nbr] = node_nbr_str

    tetrad_txt += "\n\n"
    tetrad_txt += "Graph Edges:\n"

    idx = 1
    for node, nbr_dict in graph_edge_dict.items():
        for nbr, edge_str in nbr_dict.items():
            tetrad_txt += f"{idx}. {node} {edge_str} {nbr}\n"
            idx += 1

    with open(filename, "w") as fout:
        fout.write(tetrad_txt)
    return tetrad_txt
