[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/pywhy-graphs/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/pywhy-graphs/tree/main)
[![unit-tests](https://github.com/py-why/pywhy-graphs/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/pywhy-graphs/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/pywhy-graphs/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/pywhy-graphs)
[![PyPI Download count](https://pepy.tech/badge/pywhy-graphs)](https://pepy.tech/project/pywhy-graphs)
[![Latest PyPI release](https://img.shields.io/pypi/v/pywhy-graphs.svg)](https://pypi.org/project/pywhy-graphs/)

# PyWhy-Graphs

pywhy-graphs is a Python graph library that extends [networkx](https://github.com/networkx/networkx) with the notion of a `MixedEdgeGraph` to implement a light-weight API for causal graphical structures that contain mixed-edges and contain causal graph traversal algorithms.

## Why?

Representation of causal graphical models in Python are severely lacking.

PyWhy-Graphs implements a graphical API layer for representing commmon graphs in causal inference: ADMG, CPDAG and PAG. For causal DAGs, we recommend using the `networkx.DiGraph` class and
ensuring acylicity via `networkx.is_directed_acyclic_graph` function.

Existing packages that aim to represent causal graphs either break from the networkX API, or only implement a subset of the relevant causal graphs. By keeping in-line with the robust NetworkX API, we aim to ensure a consistent user experience and a gentle introduction to causal graphical models. A `MixedEdgeGraph` instance is a composition of networkx graphs and has a similar API, with the additional notion of an "edge type", which specifies what edge type subgraph any function should operate over. For example:

```Python
# adds a directed edge from x to y
G.add_edge('x', 'y', edge_type='directed')

# adds a bidirected edge from x to y
G.add_edge('x', 'y', edge_type='bidirected')
```

Moreover, sampling from causal models is non-trivial, but a requirement for benchmarking many causal algorithms in discovery, ID, estimation and more. We aim to provide simulation modules that are easily connected with causal graphs to provide a simple robust API for modeling causal graphs and then simulating data.

# Documentation

See the [development version documentation](https://py-why.github.io/pywhy-graphs/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/pywhy-graphs/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](https://py-why.github.io/pywhy-graphs/dev/installation.html) for full details.

## Dependencies

We aim to provide a very light-weight dependency structure. Minimally, pywhy-graphs requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * networkx

Additional functionality may be required when running unit-tests and documentation.

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install pywhy-graphs is using `pip`:

    pip install pywhy-graphs

or you can add it via poetry

    poetry add pywhy-graphs

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # for vizualizing graph functionality
    poetry install --extras viz

    # if you would like an editable install of dodiscover for dev purposes
    pip install -e .

    pip install https://api.github.com/repos/py-why/pywhy-graphs/zipball/main

# Contributing

Pywhy-Graphs is always looking for new contributors to help make the package better, whether it is algorithms, documentation, examples of graph usage, and more! Contributing to Pywhy-Graphs will be rewarding because you will contribute to a much needed package for causal inference.

See our [contributing guide](https://github.com/py-why/pywhy-graphs/blob/main/CONTRIBUTING.md) for more details.

# Citing

Please refer to the Github Citation to cite the repository.
