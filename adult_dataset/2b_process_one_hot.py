"""
Read in adult_clean.csv and apply one-hot encoding to ordinal variables: employment type, occupation and race.
"""

import pandas as pd
import json

a = pd.read_csv("adult_clean.csv", index_col=0)
with open("adult_categories.json", "r") as f: categories = json.load(f)

vars = ['employment-type', 'occupation', 'race']
a = pd.get_dummies(a, columns=vars, drop_first=False)

# Use category names to rename.
for var in vars:
    for i, c in enumerate(categories[var]):
        a = a.rename(columns={f"{var}_{i}": f"{var}_{c}"})

# Write out dataset.
a.to_csv("adult_clean_one_hot.csv", index=True)