[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/py-why/pywhy-graphs/tree/main.svg?style=svg)](https://circleci.com/gh/py-why/pywhy-graphs/tree/main)
[![unit-tests](https://github.com/py-why/pywhy-graphs/actions/workflows/main.yml/badge.svg)](https://github.com/py-why/pywhy-graphs/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/py-why/pywhy-graphs/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/py-why/pywhy-graphs)

# PyWhy-Graphs (Experimental)

pywhy-graphs is a Python graph library that extends `MixedEdgeGraph` in [networkx](https://github.com/networkx/networkx) to implement a light-weight API for causal graphical structures.

Note: The API is subject to change without deprecation cycles due to the current work-in-progress `MixedEdgeGraph` class in networkx. For more information, follow the PR at https://github.com/networkx/networkx/pull/5947

## Why?

Representation of causal graphical models in Python are severely lacking.

PyWhy-Graphs implements a graphical API layer for ADMG, CPDAG and PAG. For causal DAGs, we recommend using the `networkx.DiGraph` class and
ensuring acylicity via `networkx.is_directed_acyclic_graph` function.

Existing packages that aim to represent causal graphs either break from the networkX API, or only implement a subset of the relevant causal graphs. By keeping in-line with the robust NetworkX API, we aim to ensure a consistent user experience and a gentle introduction to causal graphical models.

Moreover, sampling from causal models is non-trivial, but a requirement for benchmarking many causal algorithms in discovery, ID, estimation and more. We aim to provide simulation modules that are easily connected with causal graphs to provide a simple robust API for modeling causal graphs and then simulating data.

# Documentation

See the [development version documentation](https://py-why.github.io/pywhy-graphs/dev/index.html).

Or see [stable version documentation](https://py-why.github.io/pywhy-graphs/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, pywhy-graphs requires:

    * Python (>=3.8)
    * numpy
    * scipy
    * networkx

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install pywhy-graphs is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U pywhy-graphs

To install the package from github, clone the repository and then `cd` into the directory. You can then use `poetry` to install:

    poetry install

    # for vizualizing graph functionality
    poetry install --extras viz

    # if you would like an editable install of dodiscover for dev purposes
    pip install -e .

    pip install https://api.github.com/repos/py-why/pywhy-graphs/zipball/main
