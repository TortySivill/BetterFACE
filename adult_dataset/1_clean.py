"""
Raw data from https://archive.ics.uci.edu/ml/datasets/adult
Cleaning code adapted from https://ryanwingate.com/projects/machine-learning-data-prep/adult/adult-cleaning/.
"""

import pandas as pd
import json 

categories = {}

# Read in the data, concatenating the training set "adult.data" and test set "adult.test".
adult_data_path = 'raw/adult.data'
adult_test_path = 'raw/adult.test'
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain', 'capital-loss',
        'hours-per-week', 'native-country','compensation']
a = (pd.read_csv(adult_data_path, names=cols, sep=', ', engine='python')
     .append(pd.read_csv(adult_test_path, skiprows=1, names=cols, sep=', ', engine='python')))

print("Shape before cleaning =", a.shape)

# Fix stray decimal points in compensation.
a = a.replace({'<=50K.': '<=50K', '>50K.': '>50K'})

# Drop entries where workclass and occupation are unknown, and where workclass is Without-pay.
a = (a[(a['workclass']!='?')&
       (a['occupation']!='?')&
       (a['workclass']!='Without-pay')]
     .reset_index(drop=True))
a['idx'] = a.index

# Map the very small Armed-Forces category of occupation to Protective-serv.
a.loc[a['occupation']=='Armed-Forces','occupation'] = 'Protective-serv'

# Map Ages, Education, Workclass, and Weekly-Hours to smaller category set.
a.loc[a['workclass'].isin(['State-gov', 'Federal-gov', 'Local-gov']), 
      'employment-type'] = 'Government'
a.loc[a['workclass'].isin(['Self-emp-not-inc', 'Self-emp-inc']),      
      'employment-type'] = 'Self-Employed'
a.loc[a['workclass'].isin(['Private']),                               
      'employment-type'] = 'Privately-Employed'

# Rename weekly hours column.
a = a.rename(columns={'hours-per-week':'weekly-hours'})

# Subtract 1 from education-num so categories start at 0 (NOTE: education categories are already ordered).
a["education-num"] = a["education-num"] - 1
cat = a.groupby(['education','education-num']).size().reset_index().rename(columns={0:'count'}).sort_values(by='education-num')
print(cat)
categories["education-num"] = {i: c for i, c in enumerate(cat["education"])}
print(categories["education-num"])

# Convert categorical columns to numerical labels (NOTE: this results in alphanumeric order).
for var in ['sex', 'compensation', 'employment-type', 'occupation', 'race']:
       a[f'{var} raw'] = a[var]
       a[var] = a[var].astype('category')
       a[var] = a[var].cat.codes
       cat = a[[var,f'{var} raw','idx']].groupby([var, f'{var} raw']).count().rename(columns={'idx':'count'}).reset_index(drop=False)
       print(cat)
       categories[var] = {i: c for i, c in enumerate(cat[f'{var} raw'])}
       print(categories[var])

# Remove redundant columns.
a = a[['age',
       'employment-type',
       'education-num',
       'occupation',
       'race',
       'sex',
       'weekly-hours',
       'compensation']].copy()

# Move compensation to the end. 
_c = a.pop('compensation') 
a['compensation'] = _c

print("Shape after cleaning", a.shape)

# Write out dataset and category index as a JSON file.
a.to_csv("adult_clean.csv", index=True)
with open("adult_categories.json", "w") as f: json.dump(categories, f, indent=4)