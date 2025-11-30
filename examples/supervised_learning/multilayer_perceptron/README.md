# Multilayer Perceptron (MLP)

## Overview
A Multilayer Perceptron (MLP) is a supervised machine learning model capable of learning non-linear relationships for both classification and regression tasks.  
It consists of layers of interconnected neurons that transform input data through learned weights and activation functions.

## How It Works
1. Input features are passed into the network.  
2. Each hidden layer computes:
   - A linear combination: $z = W x + b$
   - A non-linear activation (e.g., ReLU, sigmoid, tanh)  
3. The final output layer produces:
   - Class probabilities (classification)  
   - A continuous value (regression)  
4. Errors are computed and weights are updated via backpropagation and gradient descent.

## Strengths
- Learns complex, non-linear decision boundaries  
- Highly flexible: customizable depth, width, and activations  
- Works well on many real-world datasets  

## Weaknesses
- Requires tuning many hyperparameters  
- Can overfit without regularization  
- Less interpretable than simpler models  

## Best Practices
- Normalize or standardize input features  
- Use validation data for tuning architecture and learning rate  
- Apply regularization (L2, dropout, early stopping)  