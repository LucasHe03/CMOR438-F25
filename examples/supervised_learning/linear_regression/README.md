# Linear Regression

## Overview
Linear Regression is a supervised machine learning algorithm used for predicting a continuous target variable based on one or more input features.  

It models the relationship between the inputs and the output as a **linear equation**:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

Where:  
- $y$ is the target variable  
- $x_i$ are input features  
- $\beta_i$ are coefficients  
- $\epsilon$ is the error term

## How It Works
1. Compute the coefficients ($\beta$) that minimize the **sum of squared errors** between predicted and actual values.  
2. Use the linear equation to make predictions on new data.  

## Strengths
- Simple and easy to interpret  
- Computationally efficient  
- Provides insight into feature importance via coefficients  

## Weaknesses
- Assumes a linear relationship between features and target  
- Sensitive to outliers
- Can underperform if features are correlated

## Best Practices
- Standardize or normalize features when they have different scales  
- Check for multicollinearity among features  
- Inspect residuals to validate model assumptions  