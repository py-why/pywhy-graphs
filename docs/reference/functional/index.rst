.. _functional-causal-graphical-models:

**********************************
Functional Causal Graphical Models
**********************************

.. automodule:: pywhy_graphs.classes
    :no-members:
    :no-inherited-members:

Pywhy-graphs provides a layer to convert imbue causal graphs with a data-generating
model. Currently, we only support linear models, but we plan to support non-linear
and we also do not support latent confounders yet.

To add a latent confounder, one can add a confounder explicitly, generate the data
and then drop the confounder varialble in the final dataset.

Linear
======

In order to represent linear functions, we imbue nodes with a set of node attributes:

- ``parent_functions``: a mapping of functions that map each node to a nested dictionary
    of parents and their corresponding weight and function that map parent values to
    values that are input to the node value with the weight. 
- ``gaussian_noise_function``: a dictionary with keys ``mean`` and ``std`` that
    encodes the data-generating function for the Gaussian noise.
    
    For example, if the node
    is :math:`X` and its parents are :math:`Y` and :math:`Z`, then ``parent_functions``
    and ``gaussian_noise_function`` for node :math:`X` is:
    
    .. code-block:: python
        {
            'X': {
                'parent_functions': {
                    'Y': {
                        'weight': <weight of Y added to X>,
                        'func': <function that takes input Y>,
                    },
                    'Z': {
                        'weight': <weight of Z added to X>,
                        'func': <function that takes input Z>,
                    },
                },
                'gaussian_noise_function': {
                    'mean': <mean of gaussian noise added to X>,
                    'std': <std of gaussian noise added to X>,
                }
            }
        }

:mod:`pywhy_graphs.functional.linear`: Linear functional graphs
================================================================
.. currentmodule:: pywhy_graphs.functional.linear
    
.. autosummary::
   :toctree: ../../generated/

   make_graph_linear_gaussian
   apply_linear_soft_intervention

Multidomain
===========

Currently, this submodule only supports linear functions.

Multiple-domain causal graphs are represented by selection diagrams :footcite:`bareinboim_causal_2016`,
or augmented selection diagrams (TODO: CITATION FOR LEARNING SEL DIAGRAMS).

In order to represent multidomain functions, we imbue nodes with a set of node attributes
in addition to the ones for linear functions. The nodes that are imbued with extra attributes
are the direct children of an S-node.

- ``invariant_domains``: a list of domain IDs that are invariant for this node.
- ``domain_gaussian_noise_function``: a dictionary with keys ``mean`` and ``std`` that
    encodes the data-generating function for the Gaussian noise for each non-invariant domain.
    
    .. code-block:: python
        {
            'X': {
                'domain_gaussian_noise_function': {
                    <domain_id>: {
                        'mean': <mean of gaussian noise added to X>,
                        'std': <std of gaussian noise added to X>,
                    },
                'invariant_domains': [<domain_id>, ...],
                }
            }
        }

:mod:`pywhy_graphs.functional.multidomain`: Linear functional selection diagrams
================================================================================
.. currentmodule:: pywhy_graphs.functional.multidomain
    
.. autosummary::
   :toctree: ../../generated/

   make_graph_multidomain
   find_connected_pairs
