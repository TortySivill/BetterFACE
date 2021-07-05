from collections.abc import Callable
from baseface import BaseFACE
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
import pandas as pd
import logging


class OurFACE(BaseFACE):

    def __init__(
            self,
            data: pd.DataFrame,
            clf,
            dist_func: Callable,
            rule_base: dict,
            theta: set,
            pred_threshold: float = None,
            bidirectional: bool = True,
            dist_threshold: float = 1):
        self.dist_func = dist_func
        self.rule_base = rule_base
        self.theta = theta
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
            XA: Pandas DataFrame containing NA samples.
            XB: Pandas DataFrame containing NB samples.

        Returns:
            binary matrix of size len(XA) * len(XB)
        """
        logging.info(' Creating actionability matrix from rule base.')    
        XA = XA.values
        XB = XB.values
        # actionability_matrix = cdist(XA.values, XB.values, lambda x0, x1: self._apply_rules(x0, x1)).astype(bool)
        # np.fill_diagonal(actionability_matrix, False) # No "self-loop" edges.
        return np.array([[self._apply_rules(XA[i], XB[j])
                          if i != j else False # No "self-loop" edges.
                          for j in range(len(XB))] for i in range(len(XA))])

    def _weight_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame,
            mask: np.ndarray,
            placeholder=np.inf
        ):
        """
        Pairwise distance between sample sets XA, XB according to dist_func.

        Args: 
            XA:     Pandas DataFrame containing NA samples.
            XB:     Optional second DataFrame containing NB samples.
                        NOTE: If XB is None, implicitly use XB = XA, NB = NA.
            mask:   Optional NA x NB array indicating whether to complete (True) or skip (False) each pair.
                        NOTE: If XB is None, calculation is done for pair i,j if mask[i,j] = True *OR* mask[i,j] = True.

        Returns:
            dist:   NA x NB array of distances.
            
        """
        XA = XA.values
        if len(XA.shape) == 1: XA = XA.reshape(1,-1)
        NA, dA = XA.shape; indicesA = np.array(range(NA)).reshape(-1,1)
        if XB is None: 
            NB, dB = NA, dA
            indices = (indicesA,)
            func = lambda *args: squareform(pdist(*args))
        else:       
            XB = XB.values
            if len(XB.shape) == 1: XB = XB.reshape(1,-1) 
            NB, dB = XB.shape
            indices = (indicesA, np.array(range(NB)).reshape(-1,1))
            func = cdist
        assert dA == dB
        if mask is None: mask = np.ones((NA, NB), dtype=bool)
        else: assert mask.dtype == bool and mask.shape == (NA, NB)
        if XB is None: XB = XA; mask = np.logical_or(mask, mask.T) # Apply or operation to mask if XB is None.
        else: _mask = mask
        dist = func(*indices, # Using indices instead of samples allows masking.
                    lambda i, j: self.dist_func(XA[i[0]], XB[j[0]])
                    if _mask[i,j] else placeholder)
        dist[~mask] = placeholder # Reapply mask to pairs whose mirror has been computed.
        return np.squeeze(dist)

    def _apply_rules(
            self, 
            x0, 
            x1, 
            track=False
        ):
        """Check two points against rule base.

        Args:
            x0, x1: Two input points. 
            track:  (Optional) whether to additionally track which rules are fired. 

        Returns:
            False if any rule is fired, True otherwise (if track: set of fired rules).
        """
        if track: # Assemble a set of all fired rules.
            print(x0)
            print(x1)
            fired = {k for k in self.theta if self.rule_base[k](x0, x1)}
            return len(fired) == 0, fired 
        else: # Don't track, and break as soon as *any* rule is fired.
            for k in self.theta: 
                if self.rule_base[k](x0, x1): return False
            return True