.. _simulation:

*****************
Simulation module
*****************

We provide functions for simulating structural causal models starting from a
causal graph. This is useful for testing causal discovery algorithms, which assume
an underlying graph exists and then data is generated faithful to that graph.


Time-series simulations
=======================

.. automodule:: pywhy_graphs.simulate
.. autosummary::

   simulate_linear_var_process
   simulate_data_from_var
   simulate_var_process_from_summary_graph
