import dynetx as dn 
import networkx as nx


class TimeSeriesMixedEdgeGraph:
    def __init__(self, graphs=None, ) -> None:
        pass


class TimeSeriesGraph(nx.MixedEdgeGraph):
    def __init__(self, incoming_directed_edges=None,
            incoming_bidirected_edges=None,
            incoming_undirected_edges=None, 
            directed_edge_name='directed',
            bidirected_edge_name='bidirected',
            undirected_edge_name='undirected',
            **attr):
        super().__init__(**attr)
        
        self.add_edge_type(dn.DynDiGraph(incoming_directed_edges), directed_edge_name)
        self.add_edge_type(dn.DynGraph(incoming_bidirected_edges), bidirected_edge_name)
        self.add_edge_type(dn.DynGraph(incoming_undirected_edges), undirected_edge_name)

        self._directed_name = directed_edge_name
        self._bidirected_name = bidirected_edge_name
        self._undirected_name = undirected_edge_name

    def nodes_iter(self, edge_type=None, t=None, data=False):
        pass

    def nodes(self, edge_type=None, t=None, data=False):
        if data:
            return [(k, v) for k, v in self.nodes_iter(edge_type=edge_type, t=t, data=data).items()]
        else:
            return [k for k in self.nodes_iter(edge_type=edge_type, t=t, data=data)]

    def add_interaction(self, u, v, t):
        pass

    def add_interactions_from(self, ebunch, t):
        pass

    def time_slice(self, t_from, t_to):
        pass 

    