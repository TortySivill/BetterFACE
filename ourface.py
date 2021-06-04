from collections.abc import Callable
from baseface import BaseFACE
import numpy as np
import pandas as pd
import logging


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
        features = list(XA)
        XA = XA.values
        XB = XB.values
        permission_matrix = np.ones((len(XA), len(XB)), dtype=bool)
        for i in range(len(XA)):
            for j in range(len(XB)):
                permission_matrix[i, j] = self.rule_base(features, XA[i], XB[j])
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
        return self.dist_func(XA.values, XB.values, threshold_matrix, placeholder=np.inf)
