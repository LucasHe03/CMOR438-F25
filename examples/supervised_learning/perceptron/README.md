# Perceptron

## Overview
A Perceptron is a basic supervised machine learning model used for binary classification.  
It consists of a single layer of weights connecting input features directly to the output, making predictions using a linear decision boundary.

## How It Works
1. Input features are multiplied by weights and summed: $z = W x + b$  
2. The result is passed through a step function to produce a binary output (0 or 1)  
3. During training:
   - The model compares predictions to true labels  
   - Updates weights using the Perceptron learning rule to reduce misclassification  

## Strengths
- Simple and fast to train  
- Provides a clear, linear decision boundary  
- Good for linearly separable datasets  

## Weaknesses
- Cannot model non-linear relationships  
- Performance drops on noisy or overlapping data  
- Limited to binary classification  

## Best Practices
- Scale or normalize input features  
- Ensure data is approximately linearly separable  
- Consider multi-layer perceptrons or kernel methods for complex datasets  