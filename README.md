# Machine Learning Library

**Author:** Lucas He  
**Course:** CMOR 438 - Data Science and Machine Learning  
**Semester:** Fall 2025 

--- 

## Overview
Rice2025 is a Python library implementing fundamental machine learning algorithms.

Implemented algorithms include: 
- Perceptron
- Multi-layer Perceptron
- Logistic Regression
- Linear Regression
- K-Nearest Neighbors
- Decision Trees & Regression Trees
- Random Forests
- Ensemble Methods

---

## Installation
```bash
git clone https://github.com/LucasHe03/CMOR438-F25.git
cd CMOR438-F25
pip install -e .
```

--- 

## Project Structure
```
.  
├── .github                             # Github configuration files  
│   ├── ISSUE_TEMPLATE                  # Feature request templates  
│   └── workflows                       # CI setup  
├── LICENSE                             # License information  
├── README.md                           # Project documentation (this file)  
├── examples                            # Example useage of modules  
│   └── supervised_learning  
├── src                                 # Main library source  
│   └── rice2025  
│       ├── supervised_learning         # Supervised learning algorithms  
│       └── utilities                   # Helper functions for pre/post-processing and metrics  
└── tests                               # Automated testing suite  
    ├── integration                     # Integration tests across components  
    └── unit                            # Unit tests for each module  
        ├── supervised_learning  
        └── utilities  
```

--- 

## Testing
Run all unit and integration tests using:
```bash
pytest
```

or for a specific module:
```bash
pytest /tests/unit/supervised_learning/test_knn.py
```

Continuous integration tests are configured via .github/workflows/ci.yml.