"""
===========================
On PAGs and their validity
===========================

A PAG or a Partial Ancestral Graph is a type of mixed edge
graph that can represent, in a single graph, the causal relationship
between several nodes as defined by an equivalence class of MAGs.


PAGs model this relationship by displaying all common edge marks shared 
by all members in the equivalence class and displaying circles for those marks
that are not common.

More details on inducing paths can be found at :footcite:`Zhang2008`.

"""

import pywhy_graphs
from pywhy_graphs.viz import draw
from pywhy_graphs import PAG

try:
    from dodiscover import FCI, make_context
    from dodiscover.ci import Oracle
    from dodiscover.constraint.utils import dummy_sample
except ImportError as e:
    raise ImportError(
        "The 'dodiscover' package is required to convert a MAG to a PAG."
    )


# %%
# PAGs in pywhy-graphs
# ---------------------------
# Constructing a PAG in pywhy-graphs is an easy task since
# the library provides a separate class for this purpose.
# True to the definition of PAGs, the class can contain
# directed edges, bidirected edges, undirected edges and
# cicle edges. To illustrate this, we construct an example PAG:

pag = PAG()
pag.add_edge("A", "B", pag.directed_edge_name)
pag.add_edge("B", "A", pag.circle_edge_name)
pag.add_edge("B", "C", pag.circle_edge_name)
pag.add_edge("C", "B", pag.directed_edge_name)
pag.add_edge("A", "D", pag.circle_edge_name)
pag.add_edge("D", "A", pag.circle_edge_name)


# Finally, the graph looks like this:
dot_graph = draw(pag)
dot_graph.render(outfile="new_pag.png", view=True)


# %%
# Validity of a PAG
# ---------------------------
# To check if the constructed PAG is a valid one, we 
# can simply do:


# returns True
print(pywhy_graphs.valid_pag(pag))


# %%
# References
# ----------
# .. footbibliography::