# Random Forest

## Overview
Random Forest is an ensemble supervised learning method used for both classification and regression tasks.  
It constructs multiple decision trees during training and outputs either the majority vote (classification) or **average prediction** (regression) of the individual trees.

## How It Works
1. Multiple bootstrap samples are drawn from the training data.  
2. For each sample, a decision tree is grown:
   - At each split, a random subset of features is considered.  
   - Trees are grown fully or until a stopping criterion is met.  
3. Predictions from all trees are aggregated:
   - Classification: majority vote  
   - Regression: mean of outputs  

## Strengths
- High accuracy and robustness to overfitting  
- Handles large feature spaces and missing data well  
- Provides feature importance metrics  

## Weaknesses
- Can be computationally expensive for many trees  
- Less interpretable than a single decision tree  
- Can overfit on noisy data if trees are too deep  

## Best Practices
- Tune the number of trees and maximum tree depth  
- Use feature scaling if combining with distance-based methods  