# Principal Component Analysis (PCA)

## Overview
Principal Component Analysis (PCA) is an unsupervised dimension reduction technique that transforms data into a lower-dimension space while preserving maximum variance. 

## How It Works
1. Center (and optionally scale) the data so each feature has mean zero.  
2. Compute the covariance matrix of the features.  
3. Find the eigenvectors and eigenvalues of the covariance matrix.  
4. Project the original data onto the top eigenvectors that explain the most variance.

## Strengths
- Reduces dimensionality while retaining important structure  
- Removes correlated and redundant features  
- Improves visualization and can speed up learning algorithms  

## Weaknesses
- Principal components can be hard to interpret  
- Assumes linear relationships in the data  
- Important information may be lost if too few components are kept  

## Best Practices
- Standardize features before applying PCA  
- Choose the number of components based on explained variance  
- Use PCA as a preprocessing step before clustering or classification