"""
.. _ex-draw-graphs:

=============================================================
Drawing graphs and setting their layout for visual comparison
=============================================================

One can draw a graph without setting the `pos` argument, in that case graphviz will choose how to place the nodes.
See https://graphviz.readthedocs.io/en/stable/api.html?highlight=render#graphviz.Digraph.render

If one sets the `pos` argument, the positions will be fixed.

This examples shows how to create a position layout for all the nodes (using networkx)
and pass this to other graphs so that the nodes positions are the same for the nodes with the same labels
"""

import pywhy_graphs
import networkx as nx

from pywhy_graphs import CPDAG, PAG
from pywhy_graphs.viz import draw

# create some dummy graphs: G, admg, cpdag, and pag
# this code is borrowed from the other example: intro_causal_graphs.py ;)
G = nx.DiGraph([("x", "y"), ("z", "y"), ("z", "w"), ("xy", "x"), ("xy", "y")])
admg = pywhy_graphs.set_nodes_as_latent_confounders(G, ["xy"])
cpdag = CPDAG()
cpdag.add_edges_from(G.edges, cpdag.undirected_edge_name)
cpdag.orient_uncertain_edge("x", "y")
cpdag.orient_uncertain_edge("xy", "y")
cpdag.orient_uncertain_edge("z", "y")
pag = PAG()
pag.add_edges_from(G.edges, cpdag.undirected_edge_name)

# get the layout position for the graph G using networkx
# https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
pos_G = nx.spring_layout(G, k=10)

# draw the graphs (i.e., generate a graphviz object that can be rendered)
# each time we call draw() we pass the layout position of G
dot_G = draw(G, pos=pos_G)
dot_admg = draw(admg, pos=pos_G)
dot_cpdag = draw(cpdag, pos=pos_G)
dot_pag = draw(pag, pos=pos_G)

# render the graphs using graphviz render() function
# https://graphviz.readthedocs.io/en/stable/api.html?highlight=render#graphviz.Digraph.render
dot_G.render(outfile="G.png", view=True, engine='neato')
dot_admg.render(outfile="admg.png", view=True, engine='neato')
dot_cpdag.render(outfile="cpdag.png", view=True, engine='neato')
dot_pag.render(outfile="pag.png", view=True, engine='neato')
