# Logistic Regression

## Overview
**Logistic Regression** is a supervised machine learning algorithm used for **classification tasks**.  
Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities of class membership and outputs class labels.

## How It Works
1. Compute a weighted sum of the input features: $z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$  
2. Apply the sigmoid function to obtain a probability $P(y=1 \mid X)$  
3. Convert probability to a class label using a preset threshold

## Strengths
- Simple and interpretable  
- Efficient and fast to train  
- Probabilistic output allows ranking/confidence estimation  

## Weaknesses
- Assumes a linear decision boundary
- Sensitive to outliers
- Can be affected by multicollinearity among features  

## Best Practices
- Standardize or normalize features  
- Check for multicollinearity  
- Use regularization (L1/L2) to improve generalization  