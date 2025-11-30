# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Overview
DBSCAN is an unsupervised clustering algorithm that groups together points that are closely packed while marking points in low-density regions as noise.  

## How It Works
1. Core Points: A point is a core point if at least `min_samples` neighbors fall within distance `eps`.  
2. Density Reachability:  
   - A point directly reachable from a core point is part of the same cluster.  
   - Continuing this reachability expands the cluster.  
3. Noise Points: Points not reachable from any core point are labeled as outliers.

## Strengths
- Automatically identifies the number of clusters  
- Can find arbitrarily shaped clusters  
- Works well when clusters have different shapes and sizes

## Weaknesses
- Performance degrades in high-dimensional spaces  
- Choosing good values for `eps` and `min_samples` can be tricky  
- Struggles when cluster densities vary significantly

## Best Practices
- Use a **k-distance plot** to tune `eps`  
- Standardize or normalize features before clustering  
- Reduce dimensionality (e.g., PCA) when dealing with high-dimensional data  