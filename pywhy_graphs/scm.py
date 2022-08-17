import inspect
from collections import defaultdict
from typing import Callable, Dict

import pandas as pd

from pywhy_graphs import ADMG


class StructuralCausalModel:
    """Structural Causal Model (SCM) class.

    Assumes that all exogenous variables are independent of
    each other. That is no exogenous variable is a function
    of other exogenous variables passed in.

    This assumes the causal independence mechanism, where all
    exogenous variables are independent of each other.

    Parameters
    ----------
    exogenous : Dict of functions
        The exogenous variables and their functional form
        passed in as values. This forms a symbolic mapping
        from exogenous variable names to their distribution.
        The exogenous variable functions should not have
        any parameters.
    endogenous : Dict of lambda functions
        The endogenous variable functions may have parameters.

    Attributes
    ----------
    causal_dependencies : dict
        A mapping of each variable and its causal dependencies based
        on the SCM functions.
    var_list : list
        The list of variable names in the SCM.
    _symbolic_runtime : dict
        The mapping from each variable in the SCM to the sampled
        value of that variable. Used when sampling from the SCM.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState()
    >>> func_uxy = rng.uniform
    >>> func_uz = rng.uniform
    >>> func_x = lambda u_xy: 2 * u_xy
    >>> func_y = lambda x, u_xy: x
    >>> func_z = lambda u_z: u_z**2
    >>> scm = StructuralCausalModel(
            exogenous={
                "u_xy": func_uxy,
            },
            endogenous={"x": func_x, "y": func_y},
        )

    """

    _symbolic_runtime: Dict[str, float]

    def __init__(self, exogenous: Dict[str, Callable], endogenous: Dict[str, Callable]) -> None:
        self._symbolic_runtime = dict()

        # construct symbolic table of all variables
        self.exogenous = exogenous
        self.endogenous = endogenous

        # keep track of all variables for error checking
        endog_var_list = list(endogenous.keys())
        exog_var_list = list(self.exogenous.keys())
        var_list = []
        var_list.extend(endog_var_list)
        var_list.extend(exog_var_list)
        self.var_list = var_list

        self.causal_dependencies = {}

        input_warn_list = []
        for endog_var, endog_func in endogenous.items():
            # get all variable names from endogenous function
            endog_input_vars = inspect.getfullargspec(endog_func).args

            # check for input arguments for functions that are not
            # defined within the SCM
            if any(name not in var_list for name in endog_input_vars):
                input_warn_list.extend(endog_input_vars)

            self.causal_dependencies[endog_var] = endog_input_vars

        # All variables should be defined if they have a functional form
        if input_warn_list:
            raise ValueError(
                f"Endogenous functions define a list of variables not "
                f"within the set of passed variables: {input_warn_list}"
            )

    def __str__(self):
        """For printing."""
        return (
            f"Structural Causal Model:\n"
            f"endogenous: {list(self.endogenous.keys())}\n"
            f"exogenous: {list(self.exogenous.keys())}\n"
        )

    def sample(self, n: int = 1000, include_latents: bool = True) -> pd.DataFrame:
        """Sample from the SCM.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate, by default 1000.
        include_latents : bool, optional
            Whether to include latent variables in the returned
            dataset, by default True.

        Returns
        -------
        result_df : pd.DataFrame
            The sampled dataset.
        """
        df_values = []

        # construct truth-table based on the SCM
        for _ in range(n):
            self._symbolic_runtime = dict()

            # sample all latent variables, which are independent
            for exog, exog_func in self.exogenous.items():
                self._symbolic_runtime[exog] = exog_func()

            # sample now all observed variables
            for endog, endog_func in self.endogenous.items():
                endog_value = self._sample_function(endog_func, self._symbolic_runtime)

                if endog not in self._symbolic_runtime:
                    self._symbolic_runtime[endog] = endog_value

            # add each sample to
            df_values.append(self._symbolic_runtime)

        # now convert the final sample to a dataframe
        result_df = pd.DataFrame(df_values)

        if not include_latents:
            # remove latent variable columns
            result_df.drop(self.exogenous.keys(), axis=1, inplace=True)
        else:
            # make sure to order the columns with latents first
            def key(x):
                return x not in self.exogenous.keys()

            result_df = result_df[sorted(result_df, key=key)]
        return result_df

    def _sample_function(self, func: Callable, result_table: Dict[str, float]):
        # get all input variables for the function
        input_vars = inspect.getfullargspec(func).args

        # recursive tree stopping condition
        if all(name in result_table for name in input_vars):
            return func(*[result_table[name] for name in input_vars])

        # get all variable names that we still need to sample
        # then recursively call function to sample all variables
        to_sample_vars = [name for name in input_vars if name not in result_table]

        for name in to_sample_vars:
            result_table[name] = self._sample_function(self.endogenous[name], result_table)
        return func(*[result_table[name] for name in input_vars])

    def get_causal_graph(self) -> ADMG:
        """Compute the induced causal diagram.

        Returns
        -------
        G : instance of ADMG
            The causal graphical model corresponding to
            the SCM.

        """
        edge_list = []
        latent_edge_dict = defaultdict(set)

        # form the edge lists
        for end_var, end_input_vars in self.causal_dependencies.items():
            # for every input variable, form either an edge, or a latent edge
            for input_var in end_input_vars:
                if input_var in self.endogenous:
                    edge_list.append((input_var, end_var))
                elif input_var in self.exogenous:
                    latent_edge_dict[input_var].add(end_var)

        # add latent edges
        latent_edge_list = []
        for _, pc_comps in latent_edge_dict.items():
            if len(pc_comps) == 2:
                latent_edge_list.append(pc_comps)

        G = ADMG(
            incoming_directed_edges=edge_list,
            incoming_bidirected_edges=latent_edge_list,
            name="Induced Causal Graph from SCM",
        )
        for node in self.endogenous.keys():
            G.add_node(node)

        return G
