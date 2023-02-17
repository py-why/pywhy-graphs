.. Places parent toc into the sidebar

:parenttoc: True

.. _simulation:

*****************
Simulation module
*****************

We provide functions for simulating structural causal models starting from a
causal graph. This is useful for testing causal discovery algorithms, which assume
an underlying graph exists and then data is generated faithful to that graph.


.. automodule:: pywhy_graphs.simulate
   :no-members:
   :no-inherited-members:

:mod:`pywhy_graphs.simulate`: Causal graphical model simulations
================================================================
.. currentmodule:: pywhy_graphs

.. autosummary::
   :toctree: ../../generated/

   simulate.simulate_linear_var_process
   simulate.simulate_data_from_var
   simulate.simulate_var_process_from_summary_graph
