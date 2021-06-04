# Adult dataset

Classifiers trained are logistic regression models (with default sklearn parameters) on the entire adult dataset:
- clf_raw: trained on raw dataset
- clf_one_hot: trained on dataset where nominal features have been one-hot encoded
- clf_age_occupation: train on age and one-hot encoded occupation