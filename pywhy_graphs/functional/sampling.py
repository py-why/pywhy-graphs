import numpy as np
import networkx as nx
import pandas as pd
from collections import defaultdict

from .utils import check_discrete_model


class BayesianSampling:
    def __init__(self, graph, random_state=None, n_jobs=None) -> None:
        self.graph = graph
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._topological_order = None

        self.check_model()

    def check_model(self):
        # XXX: Add check for continuous models
        if self.graph.graph.get("functional") == "discrete":
            check_discrete_model(self.graph)

    @property
    def topological_order(self):
        if self._topological_order is not None:
            return self._topological_order

        self._topological_order = list(nx.topological_sort(self.graph))

    def forward_sample(self, partial_samples=None, n_samples: int = 1):
        rng = np.random.default_rng(self.random_state)

        # check partial samples is a valid pandas dataframe
        if partial_samples is not None and not all(
            [node in self.graph.nodes for node in partial_samples.columns]
        ):
            raise ValueError(
                f"partial_samples contains columns not in the graph: {partial_samples.columns}."
            )
        if partial_samples.shape[0] != n_samples:
            raise ValueError(
                f"partial_samples has {partial_samples.shape[0]} rows, "
                f"but n_samples is {n_samples}."
            )

        # sample IID data
        sampled_data = defaultdict(list)

        # XXX: this entire for loop can be parallelized over n_samples // n_jobs
        for node in self._topological_order:
            # If values specified in partial_samples, use them. Else generate the values.
            if (partial_samples is not None) and (node in partial_samples.columns):
                sampled_data[node] = partial_samples.loc[:, node].values
            else:
                # extract the CPD for the node and the numerical states of 'node'
                cpd = self.graph.nodes[node]["cpd"]
                states = range(cpd.cardinality)

                # extract parent variables of 'node'
                evidence = cpd.variables[:0:-1]

                if len(evidence) > 0:
                    # weights are (#evidence, n_samples)
                    evidence_values = np.vstack([sampled_data[node] for node in evidence])

                    # get a map from state values to indices in the sample array
                    # and mapping each index to a weight


                    # now for each sample, we get the corresponding evidence
                    # and use it to get the weights for the sample
                    # we memoize the evidence values to avoid recomputing
                    # the same evidence values for the same sample
                    # unique_weight_indices, counts = np.unique(weight_indices, return_counts=True)
                    # samples = np.zeros((n_samples,), dtype=int)
                    # for n_samples_with_weight, weight_index in zip(counts, unique_weight_indices):
                    #     samples[weight_indices == weight_index] = np.random.choice(
                    #         states, size=n_samples_with_weight,
                    #         p=index_to_weight[weight_index]
                    #     )

                    for idx in range(n_samples):
                        # extract weights based on the conditional probability given the evidence values
                        weights = cpd.values[tuple(evidence_values[:, idx])]
                        sampled_data[node].append(rng.choice(states, p=weights))
                else:
                    # weights are (cardinality, #evidence)
                    weights = cpd.values

                    if weights.ndim != 1:
                        raise RuntimeError('wtf')

                    sampled_data[node] = rng.choice(states, size=n_samples, p=weights)

        sampled_data = pd.DataFrame(sampled_data)
        return sampled_data
