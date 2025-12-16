# rice2025 Usage Examples

This module provides examples of how to use each of the machine learning models that I have implemented in the rice2025 package.  

This README will provide a detailed analysis of each dataset used throughout the example notebooks. More general overviews are available in each of the notebooks. 

## Wine Dataset (scikit-learn)

The Wine dataset is a **multiclass classification** dataset derived from a chemical analysis of wines.  

- **Source:** `sklearn.datasets.load_wine`
- **Samples:** 178
- **Features:** 13 numeric chemical properties (e.g., alcohol, flavanoids, color intensity)
- **Classes:** 3 wine types
- **Task Type:** Classification
- **Target:** Wine label (0, 1, 2)

---

## California Housing Dataset (scikit-learn)

The California Housing dataset is a **regression** dataset based on housing data from the 1990 California census.

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Samples:** 20,640
- **Features:** 8 numeric attributes (e.g., median income, house age, average rooms)
- **Target:** Median house value (in hundreds of thousands of dollars)
- **Task Type:** Regression

---

## Breast Cancer Wisconsin Dataset (scikit-learn)

The Breast Cancer Wisconsin dataset is a **binary classification** dataset used to predict whether a tumor is malignant or benign.

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 real-valued measurements 
- **Classes:** 2 (Malignant, Benign)
- **Task Type:** Classification

---

## Zachary Karate Club Dataset (NetworkX)

The Zachary Karate Club dataset is a **social network graph** that represents friendships among members of a university karate club.

- **Source:** `networkx.karate_club_graph()`
- **Nodes:** 34 (club members)
- **Edges:** 78 social interactions
- **Graph Type:** Undirected, unweighted
- **Task Type:** Network analysis

---

## Two Moons Dataset (scikit-learn)

The Two Moons dataset is a **synthetic clustering and classification** dataset designed to illustrate non-linear decision boundaries.

- **Source:** `sklearn.datasets.make_moons`
- **Samples:** Typically 100–1,000
- **Features:** 2
- **Classes:** 2
- **Task Type:** Classification / Clustering