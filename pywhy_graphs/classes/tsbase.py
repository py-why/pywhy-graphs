import dynetx as dn
import networkx as nx


class TimeSeriesMixedEdgeGraph(nx.MixedEdgeGraph):
    def __init__(self, graphs=None, edge_types=None, **attr):
        if any(not isinstance(graph, (dn.DynDiGraph, dn.DynGraph)) for graph in graphs):
            raise RuntimeError(
                "All sub-edge-graphs of a TimeSeriesMixedEdgeGraph must"
                "be an instance of DyNetx ts graphs."
            )

        super().__init__(graphs, edge_types, **attr)

    def add_interaction(self, u, v, t=None, e=None, edge_type=None):
        if edge_type is None:
            edge_types = self.edge_types
        else:
            edge_types = [edge_type]

        for edge_type in edge_types:
            G: dn.DynGraph = self._get_internal_graph(edge_type=edge_type)
            G.add_interaction(u=u, v=v, t=t, e=e)

    def add_interactions_from(self, ebunch, t=None, e=None, edge_type=None):
        if edge_type is None:
            edge_types = self.edge_types
        else:
            edge_types = [edge_type]
            
        for edge_type in edge_types:
            G = self._get_internal_graph(edge_type=edge_type)
            G.add_interactions_from(ebunch=ebunch, t=t, e=e)

    def number_of_interactions(self, u=None, v=None, t=None, edge_type=None):
        if edge_type is None:
            edge_types = self.edge_types
        else:
            edge_types = [edge_type]

        n_interactions = 0
        for edge_type in edge_types:
            G = self._get_internal_graph(edge_type=edge_type)
            n_interactions += G.number_of_interactions(u=u, v=v, t=t, edge_type=edge_type)
    
    def has_interaction(self, u, v, t=None, edge_type=None):
        if edge_type is None:
            edge_types = self.edge_types
        else:
            edge_types = [edge_type]
        
        for edge_type in edge_types:
            G = self._get_internal_graph(edge_type=edge_type)
            has_inter = G.has_interaction(u=u, v=v, t=t, edge_type=edge_type)
            if has_inter:
                return True
        return False

    def neighbors(self, n, t=None):
        nbrs = set()
        for _, G in self.get_graphs().items():
            nbrs = nbrs.union(set(dn.all_neighbors(G, n, t=t)))
        return iter(nbrs)

    def degree(self, nbunch=None, weight=None, t=None):
        return super().degree(nbunch, weight)

    def time_slice(self, t_from, t_to):
        pass

    def interactions(self):
        pass

    def interactions_iter(self):
        pass

    def in_interactions(self):
        pass


def add_star(G: TimeSeriesMixedEdgeGraph, nodes, t: int=None):
    """Add a set of nodes that form a star.

    The first node in nodes is the middle of the star and it is connected
    to all other nodes.
    
    Parameters
    ----------
    G : TimeSeriesMixedEdgeGraph
        _description_
    nodes : iterable container
        A container of nodes.
    t : int, optional
        snapshot id, by default None.
    """
