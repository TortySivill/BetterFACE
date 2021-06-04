from scipy.spatial.distance import cdist
from collections.abc import Callable
from baseface import BaseFACE
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm


class OurFACE(BaseFACE):

    def __init__(
            self,
            data: pd.DataFrame,
            clf,
            dist_func,
            rule_base: Callable,#[[str, float, float], bool], # NOTE: This kind of type hinting doesn't work in Python 3.6.9.
            pred_threshold: float = None,
            bidirectional: bool = True,
            dist_threshold: float = 1):
        self.dist_func = dist_func
        self.rule_base = rule_base
        super().__init__(
            data,
            clf,
            pred_threshold,
            bidirectional,
            dist_threshold,
        )

    def _threshold_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame
    ) -> np.ndarray:
        """Function that determines whether to add edge to graph

        Args:
            XA: Pandas DataFrame containing (n_samples, n_features)
            XB: Pandas DataFrame containing (n_samples, n_features)

        Returns:
            binary matrix of size len(XA) * len(XB)
        """
        logging.info(' Creating permission matrix from rule base')
        permission_matrix = np.ones((len(XA), len(XB)), dtype=bool)
        for i, node_from in tqdm(enumerate(XA.index)):
            for j, node_to in enumerate(XB.index):
                if node_from != node_to:
                    permissions = []
                    for feature in XA.loc[node_from].index:
                        permissions.append(self.rule_base(feature, XA.loc[node_from][feature], XB.loc[node_to][feature]))
                    permission_matrix[i, j] = np.logical_not(np.any(np.array(permissions) == 0)).astype(int)
        return permission_matrix

    def _weight_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame,
            threshold_matrix: np.ndarray
    ):
        """Weights based on UDM

        Args:
            XA: Pandas DataFrame containing (n_samples, n_features)
            XB: Pandas DataFrame containing (n_samples, n_features)
            threshold_matrix: binary matrix of size len(XA) * len(XB)

        Returns:
            weight between points
        """
        print(XA.values.shape, XB.values.shape)
        return self.dist_func(XA.values, XB.values, threshold_matrix, placeholder=np.inf)

        # with np.errstate(divide='ignore'):
            # return cdist(XA.values, XB.values, metric='euclidean') / threshold_matrix
