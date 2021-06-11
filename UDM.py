import numpy as np

class UDM:
    """
    Implementation of Unified Distance Metric, introduced in:

    Zhang, Yiqun, and Yiu-Ming Cheung. 
    "A New Distance Metric Exploiting Heterogeneous Interattribute Relationship for Ordinal-and-Nominal-Attribute Data Clustering."
    IEEE Transactions on Cybernetics (2020).
    """
    def __init__(self, X: np.ndarray, ordinal: list): 
        """
        Initialise the metric by precomputing entropy and interdependence measures on a reference dataset X.

        Args:
            X:       Array containing N samples with d features.
            ordinal: Binary list of length d indicating whether each feature is ordinal (True) or nominal (False).

        Returns:

        """
        # === Collect measurements from dataset.===
        N, self.d = X.shape # Number of samples and features.
        categories = [np.unique(X[:,r]) for r in range(self.d)] 
        n = [len(c) for c in categories] # Number of categories per feature.
        for r in range(self.d): assert (categories[r] == range(n[r])).all(), "Categories must be 0,1,...n[r] for each r."
        Z = N*(N-1)/2 # Used for normalisation.

        # === Initialise data structures that define the distance metric. ===
        self.R = np.zeros((self.d, self.d)) # Pairwise inderdependence measure between features.
        self.psi = [np.zeros((self.d, n[r], n[r])) for r in range(self.d)] # Entropy-based distance for each feature r w.r.t. each other feature s.
        self.phi = [np.zeros((n[r], n[r])) for r in range(self.d)] # Overall distance metric for each feature.

        for r in range(self.d): # Iterate through feature pairs.
            for s in range(self.d):
 
                # === Calculate interdependence measure R. ===
                C = np.zeros((n[r], n[s]), dtype=int)
                for i in range(N): C[X[i,r], X[i,s]] += 1 # Counts for feature combinations.
                C_eq = (C * np.maximum(C-1, 0)).sum() / 2 # Number of equal-concordant sample pairs. 
                C_diff = 0 # Net difference between positive- and negative-concordant sample pairs.
                if ordinal[r] and ordinal[s]: # If both r and s are ordinal.
                    for t in range(n[r]-1):
                        Cul = 0; Cuu = C[t+1:,1:].sum() # Sums of quadrants of C matrix below current t, g.             
                        for g in range(n[s]):
                            C_diff += C[t,g] * (Cuu - Cul)                            
                            if g < n[s]-1: Cul += C[t+1:,g].sum(); Cuu -= C[t+1:,g+1].sum()
                    C_diff = abs(C_diff) # Just need absolute value of net difference.  
                else: # If at least one of r and s is nominal.
                    for t in range(n[r]):
                        for h in range(t):
                            for g in range(n[s]):
                                for u in range(g): C_diff += abs((C[t,g] * C[h,u]) - (C[t,u] * C[h,g]))                
                self.R[r,s] = (C_eq + C_diff) / Z # Final calculation to get R.     

                # === Calculate entropy-based distance psi. ===
                P = C / N # Joint probabilities found by dividing C by N.
                S_A_s = np.log2(n[s]) # Maximum-entropy normalisation term.
                if ordinal[r]: # If r is ordinal.
                    for t in range(1,n[r]): # Consider adjacent categories only.
                        P_sum = P[t] + P[t-1]
                        self.psi[r][s,t,t-1] = sum([-p * np.log2(p) for p in P_sum if p > 0]) / S_A_s # Normalised entropy of summed joint distributions.                    
                    for t in range(1,n[r]): # Fill in remaining by summation to preserve monotonicity.
                        for h in range(t-1):
                            for g in range(h, t): self.psi[r][s,t,h] += self.psi[r][s,g+1,g]
                    self.psi[r][s] += self.psi[r][s].T # Symmetric.        
                else: # If r is nominal.
                    for t in range(n[r]): # Consider all pairs of categories.
                        for h in range(t): 
                            P_sum = P[t] + P[h]
                            self.psi[r][s,t,h] = self.psi[r][s,h,t] = sum([-p * np.log2(p) for p in P_sum if p > 0]) / S_A_s 

                # === Add to overall per-feature distance phi. ===
                self.phi[r] += self.R[r,s] * self.psi[r][s] / self.d

    def __call__(self, x0, x1): 
        """
        Distance between two points according to the Unified Distance Metric.

        Args: 
            x0, x1: Two input points. 

        Returns:
            dist:   Distance.
            
        """
        assert len(x0) == len(x1) == self.d
        return np.linalg.norm([self.phi[r][x0r,x1r] for r, (x0r, x1r) in enumerate(zip(x0, x1))])