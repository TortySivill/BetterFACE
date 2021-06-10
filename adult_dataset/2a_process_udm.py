"""
Read in adult_clean.csv and apply percentile-based binning to nominal variables: age and weekly hours.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

PERCENTILES = [0,5,15,25,25,45,55,65,75,85,95,97.5,100]

bin_edges = {}

a = pd.read_csv("adult_clean.csv", index_col=0)

for var in ["age", "weekly-hours"]:
    a[f"{var}_unbinned"] = a[var]
    bin_edges[var] = list(np.unique(np.percentile(a[f"{var}_unbinned"].values, PERCENTILES)))
    a[var] = pd.cut(a[f"{var}_unbinned"], bins=bin_edges[var], include_lowest=True, labels=False)
    print(f"Bin edges for {var}: {bin_edges[var]}")
    # Plot histograms for reference.
    plt.figure(); plt.title(var)
    plt.hist(a[f"{var}_unbinned"].values)
    
# Select and reorder columns.
a = a[["age","employment-type","education-num","occupation","race","sex","weekly-hours","compensation"]]

# Write out dataset and bin edge index as a JSON file.
a.to_csv("adult_clean_udm.csv", index=True)
with open("adult_udm_bin_edges.json", "w") as f: json.dump(bin_edges, f, indent=4)

plt.show()