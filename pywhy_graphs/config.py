import os.path as op
import platform
import re
import sys
from enum import Enum, EnumMeta
from functools import partial

import numpy as np


class MetaEnum(EnumMeta):
    """Meta enumeration to make 'in' keyword work."""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

    # Prints out the name of the type
    def __str__(self):
        return self.name


class EdgeType(Enum, metaclass=MetaEnum):
    """Enumeration of different causal edge endpoints.

    Categories
    ----------
    directed : str
        Signifies arrowhead ("->") edges.
    circle : str
        Signifies a circle ("*-o") endpoint. That is an uncertain edge,
        which is either circle with directed edge (``o->``),
        circle with undirected edge (``o-``), or
        circle with circle edge (``o-o``).
    undirected : str
        Signifies an undirected ("-") edge. That is an undirected edge (``-``),
        or circle with circle edge (``-o``).

    Notes
    -----
    The possible edges between two nodes thus are:

    ->, <-, <->, o->, <-o, o-o

    In general, among all possible causal graphs, arrowheads depict
    non-descendant relationships. In DAGs, arrowheads depict direct
    causal relationships (i.e. parents/children). In ADMGs, arrowheads
    can come from directed edges, or bidirected edges
    """

    ALL = "all"
    DIRECTED = "directed"
    BIDIRECTED = "bidirected"
    CIRCLE = "circle"
    UNDIRECTED = "undirected"


# Taken from causal-learn Endpoint.py
# A typesafe enumeration of the types of endpoints that are permitted in
# Tetrad-style graphs: tail (--) null (-), arrow (->), circle (-o) and star (-*).
# 'TAIL_AND_ARROW' and 'ARROW_AND_ARROW' means there are two types of edges (<-> and -->)
# between two nodes.
# 'TAIL_AND_TAIL' means there are two types of edges with two tails ending on this endpoint
class CLearnEndpoint(Enum, metaclass=MetaEnum):
    """Enumeration of causal-learn endpoints."""

    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2
    STAR = 3
    TAIL_AND_ARROW = 4
    ARROW_AND_ARROW = 5
    TAIL_AND_TAIL = 6  # added by pywhy.


class TigramiteEndpoint(Enum, metaclass=MetaEnum):
    """Enumeration of causal-learn endpoints."""

    TAIL = "--"
    NULL = ""
    ARROW = "->"
    CIRCLE = "-o"
    STAR = "-*"
    TAIL_AND_ARROW = "+->"
    # ARROW_AND_ARROW


ARRAY_ENUMS = {
    "clearn": CLearnEndpoint,
}


def _pl(x, non_pl="", pl="s"):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (int, np.generic)) else len(x)
    return non_pl if len_x == 1 else pl


def _get_numpy_libs():
    bad_lib = "unknown linalg bindings"
    try:
        from threadpoolctl import threadpool_info
    except Exception as exc:
        return bad_lib + f" (threadpoolctl module not found: {exc})"
    pools = threadpool_info()
    rename = dict(
        openblas="OpenBLAS",
        mkl="MKL",
    )
    for pool in pools:
        if pool["internal_api"] in ("openblas", "mkl"):
            return (
                f'{rename[pool["internal_api"]]} '
                f'{pool["version"]} with '
                f'{pool["num_threads"]} thread{_pl(pool["num_threads"])}'
            )
    return bad_lib


def sys_info(fid=None, show_paths=False, *, dependencies="user"):
    """Print the system information for debugging.

    This function is useful for printing system information
    to help triage bugs.

    Parameters
    ----------
    fid : file-like | None
        The file to write to. Will be passed to :func:`print()`.
        Can be None to use :data:`sys.stdout`.
    show_paths : bool
        If True, print paths for each module.
    dependencies : 'user' | 'developer'
        Show dependencies relevant for users (default) or for developers
        (i.e., output includes additional dependencies).

    Examples
    --------
    Running this function with no arguments prints an output that is
    useful when submitting bug reports::

    >>> import pywhy_graphs
    >>> pywhy_graphs.sys_info() # doctest: +SKIP
    Platform:      Linux-4.15.0-1067-aws-x86_64-with-glibc2.2.5
    Python:        3.8.1 (default, Feb  2 2020, 08:37:37)  [GCC 8.3.0]
    Executable:    /usr/local/bin/python
    CPU:           : 36 cores
    Memory:        68.7 GB

    numpy:            1.21.5 {OpenBLAS 0.3.17 with 8 threads}
    scipy:            1.8.0
    networkx:         2.8.8

    sklearn:          1.2.0
    matplotlib:       3.6.2 {backend=MacOSX}
    pandas:           1.5.2
    pygraphviz:       Not found
    causal-learn:     Not found
    joblib:           1.2.0

    pywhy_graphs:     0.0.0
    dodiscover:       Not found
    dowhy:            0.8
    """  # noqa: E501
    # _check_option('dependencies', dependencies, ('user', 'developer'))
    ljust = 21 if dependencies == "developer" else 18
    platform_str = platform.platform()
    if platform.system() == "Darwin" and sys.version_info[:2] < (3, 8):
        # platform.platform() in Python < 3.8 doesn't call
        # platform.mac_ver() if we're on Darwin, so we don't get a nice macOS
        # version number. Therefore, let's do this manually here.
        macos_ver = platform.mac_ver()[0]
        macos_architecture = re.findall("Darwin-.*?-(.*)", platform_str)
        if macos_architecture:
            macos_architecture = macos_architecture[0]
            platform_str = f"macOS-{macos_ver}-{macos_architecture}"
        del macos_ver, macos_architecture

    out = partial(print, end="", file=fid)
    out("Platform:".ljust(ljust) + platform_str + "\n")
    out("Python:".ljust(ljust) + str(sys.version).replace("\n", " ") + "\n")
    out("Executable:".ljust(ljust) + sys.executable + "\n")
    out("CPU:".ljust(ljust) + f"{platform.processor()}: ")
    try:
        import multiprocessing
    except ImportError:
        out("number of processors unavailable " '(requires "multiprocessing" package)\n')
    else:
        out(f"{multiprocessing.cpu_count()} cores\n")
    out("Memory:".ljust(ljust))
    try:
        import psutil
    except ImportError:
        out('Unavailable (requires "psutil" package)')
    else:
        out(f"{psutil.virtual_memory().total / float(2 ** 30):0.1f} GB\n")
    out("\n")
    libs = _get_numpy_libs()
    use_mod_names = (
        "numpy",
        "scipy",
        "networkx",
        "",
        "sklearn",
        "matplotlib",
        "pandas",
        "pygraphviz",
        "causal-learn",
        # "tigramite",  # no version
        "joblib",
        "",
        "pywhy_graphs",
        "dodiscover",
        "dowhy",
    )

    if dependencies == "developer":
        use_mod_names += (
            "",
            "sphinx",
            "sphinx_gallery",
            "numpydoc",
            "pydata_sphinx_theme",
            "pytest",
            "nbclient",
            "poetry",
            "poethepoet",
        )
    for mod_name in use_mod_names:
        if mod_name == "":
            out("\n")
            continue
        out(f"{mod_name}:".ljust(ljust))
        try:
            mod = __import__(mod_name)
        except Exception:
            out("Not found\n")
        else:
            out(mod.__version__)
            if mod_name == "numpy":
                out(f" {{{libs}}}")
            elif mod_name == "matplotlib":
                out(f" {{backend={mod.get_backend()}}}")
            if show_paths:
                out(f'\n{" " * ljust}•{op.dirname(mod.__file__)}')
            out("\n")
