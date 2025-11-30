# Decision Trees

## Overview
A Decision Tree is a supervised machine learning algorithm used for both classification and regression.  

It works by recursively splitting data into smaller groups based on feature values, forming a tree structure where each internal node represents a decision rule and each leaf represents a prediction.

## How It Works
1. Start with the full dataset at the root.  
2. At each step, select the best feature and threshold that splits the data to maximize purity.  
3. Recursively repeat the splitting process on each branch.  
4. Stop when a stopping condition is met (e.g., depth limit, minimum samples).  
5. For classification: each leaf predicts the most common class.  
   For regression: each leaf predicts the average target value.

## Strengths
- Easy to visualize and interpret  
- Handles both numerical and categorical features  
- Can capture non-linear relationships  

## Weaknesses
- Prone to overfitting  
- Small changes can lead to very different trees  
- Often less accurate than ensemble methods

## Best Practices
- Use **max_depth** or **min_samples_split** to prevent overfitting  
- Consider pruning the tree after training  
- Use cross-validation to tune hyperparameters  