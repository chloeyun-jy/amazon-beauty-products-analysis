# Amazon Beauty Products: Data Analysis and Predictive Modeling

## Project Overview

This project explores Amazon Beauty product review interactions through data analysis, feature engineering, predictive modeling, and mathematical optimization.

Using a dataset of 1,348,246 review records, the project investigates user behavior, product popularity, rating patterns, and temporal trends. It also explores machine learning approaches and analytical techniques to extract insights from large-scale user-product interaction data.

The project was implemented in Python using data analysis, scientific computing, visualization, and machine learning libraries.

## Research Objectives

- Analyze user engagement and product interaction patterns.
- Examine rating distributions, product popularity, and category-level differences.
- Identify temporal trends in customer review activity.
- Engineer user-level, product-level, and category-level features.
- Explore predictive models for rating prediction and positive-rating classification.
- Apply dimensionality reduction, clustering, and mathematical optimization techniques.

## Dataset Description

The dataset contains Amazon Beauty product review interactions, with **1,348,246 review records** and six original fields.

The analysis uses five fields for data processing and modeling, while the URL field is retained for source tracking.

| Variable | Description |
|---|---|
| UserId | User identifier |
| ProductId | Product identifier |
| ProductType | Product category/type |
| Rating | User-provided product rating |
| Timestamp | Unix timestamp of the interaction |
| URL | Source-tracking field |

**Data source:** Kaggle Amazon Beauty product rating dataset.

The original CSV file is not included in this repository due to its large size. Users must obtain the dataset separately before running the notebook.

## Methodology

### 1. Data Processing and Profiling

- Load and process relevant dataset variables.
- Convert Unix timestamps into datetime values.
- Derive temporal features, including year, month, quarter, and year-month.
- Construct a binary positive-rating indicator.
- Examine user activity, product interactions, product-type coverage, rating statistics, missing values, and interaction matrix sparsity.

### 2. Exploratory Data Analysis

- Analyze rating distributions and user activity.
- Examine product interaction frequencies and popularity.
- Investigate product-category patterns.
- Explore temporal trends in review activity.
- Examine the long-tail distribution of product popularity.

### 3. Model Dataset Construction

- Apply chronological training, validation, and test partitions.
- Use minimum interaction thresholds for users and products.
- Support proportional sampling with a maximum modeling dataset size of 50,000 records to manage computational requirements.

### 4. Feature Engineering

Construct user-level, product-level, and category-level features, including:

- Interaction counts
- Average ratings
- Rating variability
- Category diversity
- Time-related variables

### 5. Predictive Modeling and Evaluation

The project explores multiple machine learning approaches:

- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)
- Logistic Regression
- Neural network approaches using PyTorch (optional)

Model evaluation uses the following metrics:

| Task | Evaluation Metrics |
|---|---|
| Regression | RMSE, MAE |
| Classification | Precision, Recall, ROC-AUC |

### 6. Additional Analytical Techniques

The project also incorporates:

- **Principal Component Analysis (PCA):** Dimensionality reduction.
- **Truncated SVD:** Dimensionality reduction for sparse interaction data.
- **K-Means Clustering:** Unsupervised grouping and pattern exploration.
- **Mixed-Integer Linear Programming (MILP):** Mathematical optimization using decision variables and linear constraints.

## Technologies

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Scientific Computing | SciPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch (optional) |
| Development Environment | Jupyter Notebook |

## Project Structure

```text
amazon-beauty-products-analysis/
│
├── Amazon Beauty Products.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The original dataset is stored separately and is not tracked by Git.

## How to Run

1. Clone or download this repository.
2. Obtain the original dataset file named `Amazon_Beauty_Recommendation.csv`.
3. Place the CSV file in the notebook's current working directory.
4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Launch Jupyter Notebook or JupyterLab.
6. Open `Amazon Beauty Products.ipynb` and execute the cells in sequence.

**Note:** The dataset is not included in this repository. The notebook requires the original CSV file to run successfully.

## Reproducibility

- A fixed random seed (`42`) is used for reproducibility.
- Chronological data partitioning is applied in the modeling workflow.
- Dependencies are listed in `requirements.txt`.
- PyTorch is optional and is required only for the corresponding neural network components.

## Author

**Chloe Yun**

