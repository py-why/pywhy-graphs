from typing import List
from pywhy_graphs.typing import Node
from .pag import PAG

class PsiPAG(PAG):
    def __init__(self, incoming_directed_edges=None, incoming_undirected_edges=None, incoming_bidirected_edges=None, incoming_circle_edges=None, directed_edge_name: str = "directed", undirected_edge_name: str = "undirected", bidirected_edge_name: str = "bidirected", circle_edge_name: str = "circle", 
        f_nodes:List[Node]=None, **attr):
        super().__init__(incoming_directed_edges, incoming_undirected_edges, incoming_bidirected_edges, incoming_circle_edges, directed_edge_name, undirected_edge_name, bidirected_edge_name, circle_edge_name, **attr)

        self.f_nodes = f_nodes