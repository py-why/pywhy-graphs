.. _functional-causal-graphical-models:

**********************************
Functional Causal Graphical Models
**********************************

.. automodule:: pywhy_graphs.functional
    :no-members:
    :no-inherited-members:

Pywhy-graphs provides a layer to convert imbue causal graphs with a data-generating
model. Currently, we only support linear models, but we plan to support non-linear
and we also do not support latent confounders yet.

To add a latent confounder, one can add a confounder explicitly, generate the data
and then drop the confounder varialble in the final dataset. In the roadmap of this submodule,
the plan is to represent any bidirected edge as a uniformly randomly distributed variable
that has an additive noise effect on both variables simulatanously.

Representing a node's functional relationships
==============================================

Within an acyclic causal diagram, each node has a set of observed parents and an exogenous parent variable.
The exogenous parent variable is the variable that is not a child of any other node in the graph and
implicitly represents all exogenous noise in the causal system that affects said node. The set of
observed parents can be the empty set. When there are observed parents, the node's value is the following
function:

    .. math:: node = f(observed\_parents, exogenous\_parent)

The causal diagram locally around ``node`` looks like :math:`observed\_parents \rightarrow node \leftarrow exogenous\_parent`,
where ``observed_parents`` can be multiple sets of direct parents. In general, this can be arbitrarily complex,
since the function ``f`` can mean anything. In our simulations, we assume additive noise,
so the node is a linear combination of possibly nonlinear functions.

    .. math:: node = f(observed\_parents) + g(exogenous\_parent)

In order to represent this function, we imbue each node with a set of node attributes:

- ``parent_function``: This computes :math:`f(observed\_parents)` for any node.
- ``exogenous_function``: This computes :math:`g(exogenous\_parent)` for any node.

Then the node value is a deterministically computed. If there are no parents, then
the node attribute will contain `None`. This enables stochasticity in the data-generating
process due to the inherent randomness that we can attach to the distribution of ``exogenous_parent``.
Due to the multivariate input nature of ``parent_function``, it must be a Callable that takes
keyword arguments of the observed parents and returns a single value. Due to the univariate input
nature of ``exogenous_function``, it must be a Callable that takes a single value and
returns a single value.

Multiple Distributions: Interventions and Domain Shifts
-------------------------------------------------------

Next, we discuss how to represent multiple distributions in a single graph. In the context of causal inference,
there are two graphical representations that allow for a general treatment of multiple distributions:
the augmented causal diagram :footcite:`dawid2002influencediagrams` and the selection diagram :footcite:`bareinboim_causal_2016`.
The augmented causal diagram is a graph that augments the original causal diagram with a set of
F-nodes that represent interventions. The selection
diagram is a graph that augments the original causal diagram with a set of S-nodes that represent
domain shifts. In both cases, the augmented graph is acyclic. They can also be combined to simulataneously
represent interventions and domain shifts.

To represent both types of distribution changes in the same graph, we note that S-nodes explicitly
either change the type of function that is used to compute the node value, or changes the distribution
of the exogenous parent variable. In the case of interventions, the function :math:`f` is changed
only.

**Interventional distribution change:**

In the interventional case, each F-node then points to any number of observed nodes in the graph.

Each observed variable node that a F-node points to has a node attribute ``intervention_functions``
that maps each of its parent F-nodes to a possibly new function that is used to compute
:math:`f'(observed\_parents)`.

**Domain change distribution change:**

In the domain shift case, each S-node has a node attribute ``domains`` that is a
unique tuple of domain integers, indicating the pair of domains that are being shifted between.
Then each node that the S-node points to has a node attribute ``domain_parent_functions`` that
maps each domain ID to a possibly new function that is used to compute :math:`f'(observed\_parents)`.
In addition, each node that the S-node points to has a node attribute ``domain_exogenous_distribution``
that maps each domain to a possibly new function that is used to compute :math:`g'(exogenous\_parent)`.

Note to sample from multiple domain changes, we always set the smallest domain ID to be the reference
distribution. For example, if domain IDs are ``(0, 1, 4, 5)``, then the reference domain is domain ``0``.

Sampling from the graph
-----------------------

Now, we have discussed how we generally represent the functional relationships of each node in the graph.
We now discuss how to sample from the graph. We first sample from the exogenous parent variable of
every observed node in the graph, which may be a function of the domain ID if the domain ID is defined.
Then, we sample the observed variables in topological order as a function of their exogenous variables
and their causal parents. The distribution sampled here is always the observational distribution of the
first domain (e.g. domain 1 out of N domains).

Given, a functional graph with multiple distributions (e.g. through interventions, or S-nodes), we
can sample the additional distributions by sampling the observed nodes in topological order again.
Consider sampling from a different domain. Each node that is a child of a S-node for the domain
that we are considering is sampled from the following distribution:

    .. math:: node = f'(observed\_parents) + g'(exogenous\_parent)

where f' and g' are new functions that are encoded in the node's ``domain_parent_functions``
and ``domain_exogenous_distribution`` dictionaries. These uniquely define the new distribution
as a result of the domain shift. 

Similarly, sampling from interventional distributions will consider each child of an F-node
for the intervention and domain that we are considering (i.e. the input should specify the
domain ID and the intervention setting we want to sample). Then we similarly sample the relevant
``domain_exogenous_distribution`` and ``intervention_functions``. Note that ``domain_parent_functions``
are not used in the interventional case, since the interventions take precedence over the domain
shift in terms of altering the functional relationship with respect to observed variables. However,
we implicitly assume the exogenous distribution is unalterable by the intervention.

Limitations
-----------

It is important to explicitly note some limitations of generating data with this API.

1. The graph must be well-defined: The graph must be acyclic and already be defined
   with a structure, before adding functional relationships.
2. The graph currently may not contain latent confounders: We plan to add this functionality
   in the future. But as of now, there is no way to represent the functional relationship
   of :math:`X \leftrightarrow Y`.
3. Additive noise: Currently, we only support additive noise. We plan to add multiplicative
   noise in the future.
4. Univariate input/output: We do not explore the possibility of a multivariate input/output
   distribution. For example, if :math:`X \in \mathbb{R}^d` and a parent of X is :math:`Y \in \mathbb{R}^m`,
   where Y is m-dimensional and X is d-dimensional and ``f`` is a function mapping Y to X, then
   this is not supported.

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

Discrete functional graphs
==========================
.. currentmodule:: pywhy_graphs.functional
    
.. autosummary::
   :toctree: ../../generated/

   make_graph_linear_gaussian


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

Linear functional graphs
========================
.. currentmodule:: pywhy_graphs.functional
    
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

Linear functional selection diagrams
====================================
.. currentmodule:: pywhy_graphs.functional
    
.. autosummary::
   :toctree: ../../generated/

   make_graph_multidomain
