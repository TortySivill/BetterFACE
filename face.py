from baseface import BaseFACE
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import logging


class FACE(BaseFACE):
    """
    Implementation of Poyiadzi et al (2020) FACE: Feasible and Actionable Counterfactual Explanations

    Attributes:
        kde: kernel-density estimator
        density_threshold: low density threshold to prune nodes from graph

    """

    def __init__(
            self,
            data: pd.DataFrame,
            clf,
            pred_threshold: float = None,
            bidirectional: bool = False,
            dist_threshold: float = 1,
            kde=gaussian_kde,
            density_threshold: float = 0.01):
        self.kde = kde(data.T)
        self.density_threshold = density_threshold
        super().__init__(
            data,
            clf,
            pred_threshold,
            bidirectional,
            dist_threshold,
        )

    def _weight_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame,
            threshold_matrix: np.ndarray
    ):
        """Weights based on kernel-density estimator

        Args:
            XA: Pandas DataFrame containing (n_samples, n_features)
            XB: Pandas DataFrame containing (n_samples, n_features)
            threshold_matrix: binary matrix of size len(XA) * len(XB)

        Returns:
            weight between points
        """
        with np.errstate(divide='ignore'):
            dist = cdist(XA.values, XB.values, metric='euclidean') / threshold_matrix
            XA_dup = np.repeat(XA.values, len(XB), axis=0)
            XB_dup = np.repeat(XB.values, len(XA), axis=0)
            weight = -np.log(self.kde((XA_dup.T + XB_dup.T) / 2).reshape(len(XA), len(XB)) * dist)
        return weight

    def prune_nodes(self):
        """removes nodes that do not meet density threshold

        Returns:

        """
        unconnected_nodes = [node for node, deg in self.G.degree if deg == 0]
        self.G.remove_nodes_from(unconnected_nodes)
        if len(unconnected_nodes) > 0:
            logging.info(f' {len(unconnected_nodes)} nodes removed as unconnected. Graph now has {len(self.G.nodes)}')

        low_density = self.data.loc[list(self.G.nodes)][self.kde(self.data.loc[list(self.G.nodes)].T)
                                                        < self.density_threshold].index
        self.G.remove_nodes_from(low_density)
        if len(low_density) > 0:
            logging.info(f' {len(low_density)} nodes removed due to low density. Graph now has {len(self.G.nodes)}')
