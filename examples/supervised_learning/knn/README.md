# K-Nearest Neighbors (KNN)

## Overview
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.  

It does not learn a model during training. Rather, it makes predictions by finding the *k* closest data points to a new input and using them to determine the output.

## How It Works
1. Choose a value for **k**.  
2. Compute the distance between the new "predict" point and all training points.  
3. Select the **k nearest neighbors**.  
4. Predict using:  
   - **Classification:** majority vote  
   - **Regression:** average of the neighbors  

## Strengths
- Very easy to understand and implement  
- No training time  
- Works well for non-linear decision boundaries  

## Weaknesses
- Slow at prediction time, as it must compare to all points
- Sensitive to outliers
- Choosing the right **k** is sometimes difficult

## Best Practices
- Normalize or standardize all features  
- Use cross-validation to pick a good **k**  
- Remove or minimize outliers  