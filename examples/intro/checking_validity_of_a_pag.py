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


try:
    from dodiscover import FCI, make_context
    from dodiscover.ci import Oracle
    from dodiscover.constraint.utils import dummy_sample
except ImportError as e:
    raise ImportError("The 'dodiscover' package is required to convert a MAG to a PAG.")


# %%
# References
# ----------
# .. footbibliography::
