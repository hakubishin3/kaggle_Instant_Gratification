# Solution

## Findings
EDA shows that the competition data was created by sklearn's make_classification with the following parameters.
  - n_clusters_per_class=3
  - hypercube=True
  - flip_y=0.05
  - n_redundant=0
  - n_repeated=0
  - class_sep=1
  - shift=0
  - scale=1

## Training
Step 1. caluculate GaussianMixture separately for each target label  
Step 2. train QDA using the cluster labels created in Step 1  
