import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("adult_dataset/adult_raw.csv")

# Subtract 1 from education-num so categories start at 0.
data["education-num"] = data["education-num"] - 1

# Bin age and weekly hours using percentiles
for var in ["age", "weekly-hours"]:
    data[f"{var}_unbinned"] = data[var]
    bin_edges = np.unique(np.percentile(data[f"{var}_unbinned"].values, [0,5,15,25,25,45,55,65,75,85,95,97.5,100]))
    data[var] = pd.cut(data[f"{var}_unbinned"], bins=bin_edges, include_lowest=True, labels=False)
    print(f"Bins for {var}: {bin_edges}")
    plt.figure(); plt.title(var)
    plt.hist(data[f"{var}_unbinned"].values)
    
# Select and reorder columns.
data = data[["age","employment-type","education-num","occupation","race","sex","weekly-hours","compensation"]]
print(data.head())
data.to_csv(f"adult_dataset/adult_udm.csv", index=False)

plt.show()