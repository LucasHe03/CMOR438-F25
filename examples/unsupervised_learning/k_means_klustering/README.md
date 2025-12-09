# K-Means Clustering

## Overview
K-Means is an unsupervised clustering algorithm that partitions data into K clusters by grouping points based on proximity to cluster centroids. Its goal is to minimize the distance between points and their assigned cluster centers.

## How It Works
1. Initialization: Choose `K` initial centroids.
2. Assignment Step: Assign each data point to the nearest centroid.
3. Update Step: Recompute each centroid as the mean of all points assigned to it.
4. Repeat: Continue assignment/update steps until centroids stop changing or the maximum number of iterations is reached.

## Strengths
- Simple to implement and very fast  
- Scales well to large datasets  
- Works well when clusters are spherical and roughly similar in size  

## Weaknesses
- Must choose `K` ahead of time  
- Sensitive to outliers and noise  
- Performance depends on initial centroid positions  

## Best Practices
- Standardize or normalize data before clustering  
- Run K-Means multiple times (`n_init`) to avoid poor initializations  
- Consider using PCA for high-dimensional datasets  