.. currentmodule:: pywhy_graphs

.. _glossary:

=========================================
Glossary of Common Terms and API Elements
=========================================

This glossary hopes to definitively represent the tacit and explicit
conventions applied in Pywhy-Graphs and its API, while providing a reference
for users and contributors. It aims to describe the concepts and either detail
their corresponding API or link to other relevant parts of the documentation
which do so. By linking to glossary entries from the API Reference and User
Guide, we may minimize redundancy and inconsistency.

We begin by listing general concepts (and any that didn't fit elsewhere), but
more specific sets of related terms are listed below:
:ref:`glossary_attributes`.

General Concepts
================

.. glossary::

    1d
    1d array
        One-dimensional array. A NumPy array whose ``.shape`` has length 1.
        A vector.

    2d
    2d array
        Two-dimensional array. A NumPy array whose ``.shape`` has length 2.
        Often represents a matrix.

    API
        Refers to both the *specific* interfaces for graphs implemented in
        pywhy-graphs and the *generalized* conventions across types of
        graphs as described in this glossary and :ref:`overviewed in the
        contributor documentation <api_overview>`.

        The specific interfaces that constitute pywhy-graphs's public API are
        largely documented in :ref:`api_ref`. However, we less formally consider
        anything as public API if none of the identifiers required to access it
        begins with ``_``.  We generally try to maintain :term:`backwards
        compatibility` for all objects in the public API.

        Private API, including functions, modules and methods beginning ``_``
        are not assured to be stable.

    callable
        A function, class or an object which implements the ``__call__``
        method; anything that returns True when the argument of `callable()
        <https://docs.python.org/3/library/functions.html#callable>`_.

    c-components
    c_components
    c components
        A set of nodes in a graph that contain a bidirected edge path between all
        nodes. Stands for "confounded components".

    docstring
        The embedded documentation for a module, class, function, etc., usually
        in code as a string at the beginning of the object's definition, and
        accessible as the object's ``__doc__`` attribute.

        We try to adhere to `PEP257
        <https://www.python.org/dev/peps/pep-0257/>`_, and follow `NumpyDoc
        conventions <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

    examples
        We try to give examples of basic usage for most functions and
        classes in the API:

        * as doctests in their docstrings (i.e. within the ``pywhy_graphs/`` library
          code itself).
        * as examples in the :ref:`example gallery <general_examples>`
          rendered (using `sphinx-gallery
          <https://sphinx-gallery.readthedocs.io/>`_) from scripts in the
          ``examples/`` directory, exemplifying key features or parameters
          of the graph/function.  These should also be referenced from the
          User Guide.
        * sometimes in the :ref:`User Guide <user_guide>` (built from ``doc/``)
          alongside a technical description of the estimator.

    experimental
        An experimental tool is already usable but its public API, such as
        default parameter values or fitted attributes, is still subject to
        change in future versions without the usual :term:`deprecation`
        warning policy.

    F-node
        A special node that is used in graphs that represents intervention
        targets. It is represented in pywhy-graphs as a pair of nodes where
        the first element is always the letter ``'F'`` and the second is
        an integer. For example, ``('F', 0)`` is an F-node.

    gallery
        See :term:`examples`.

    joblib
        A Python library (https://joblib.readthedocs.io) used in pywhy-graphs to
        facilite simple parallelism and caching.  Joblib is oriented towards
        efficiently working with numpy arrays, such as through use of
        :term:`memory mapping`. See :ref:`parallelism` for more
        information.

    lag
        The time-delay of a specific time-series graph node.

    Markov equivalence class
    equivalence class
        A graph that represents a set of graphs that preserve the same conditional
        independences.

    ``n_features``
        The number of :term:`features`.

    ``n_samples``
        The number of :term:`samples`.

    np
        A shorthand for Numpy due to the conventional import statement::

            import numpy as np

    nx
        A shorthand for Networkx due to conventional import statement::
            
            import networkx as nx

    node
        An element in a graph, similar to how Networkx defines them. Note
        this is distinctly different from a "variable" in time-series graphs.

    tsnode
        A shorthand for nodes in a time-series graph. A tsnode is defined in
        pywhy-graphs by a tuple, where the first element is the variable name
        and the second is the corresponding time-lag. For example ``('x', 0)``
        and ``('x', -1)`` are tsnodes for variable ``'x'`` and time-lags 0 and -1.

    pair
        A tuple of length two.

    pd
        A shorthand for `Pandas <https://pandas.pydata.org>`_ due to the
        conventional import statement::

            import pandas as pd

    sample
    samples
        We usually use this term as a noun to indicate a single feature vector.
        Elsewhere a sample is called an instance, data point, or observation.
        ``n_samples`` indicates the number of samples in a dataset, being the
        number of rows in a data array :term:`X`.

    SCM
    Structural Causal Model
        A model that comprises of a 4-tuple :math:`\langle V, U, P(U), F \rangle`, where
        V is the set of endogenous (observed) variables, U is the set of exogenous (latent)
        variables, P(U) is the probability distributions associated for U and F is
        the set of functions that defines each :math:`v \in V`. A SCM induces a causal
        graphical model by simply reading off the parent/children relationships in F and
        then allowing for latent confounders if any :math:`u \in U`` is shared among the
        same endogenous variables.

    sigma_map
        Only used for intervention graphs. Maps F-nodes to their distributions.

    symmetric_difference_map
        Only used for intervention graphs. Maps F-nodes to the symmetric difference
        of a pair of intervention targets. For example, if ``{'x', 'y'}`` and ``{'x'}``
        are the pair of intervention targets associated with a F-node ``('F', 0)``,
        then the symmetric difference map will map ``('F', 0)`` to ``{'y'}``.

    variable
        A set of nodes in a time-series graph corresponding to the same time-series
        component. For example ``[('x', 0), ('x', -1), ('x', -2)]`` represent
        nodes in a time-series graph that are all part of the same variable ``'x'``.