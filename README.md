# 🌸 Iris Flower Classification

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A complete, production-ready machine learning pipeline that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using their sepal and petal measurements. This project demonstrates an end-to-end ML workflow including data exploration, visualization, preprocessing, model training, evaluation, and comparison.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Data Exploration](#1-data-exploration)
  - [Visualization](#2-visualization)
  - [Preprocessing](#3-preprocessing)
  - [KNN Classifier](#4-knn-classifier)
  - [Decision Tree Classifier](#5-decision-tree-classifier)
  - [Model Comparison](#6-model-comparison)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This project implements two classic machine learning algorithms — **K-Nearest Neighbors (KNN)** and **Decision Tree** — to classify Iris flowers based on four morphological features. The entire pipeline is contained in a single, well-documented Python script with rich visualizations saved as high-resolution PNG files.

---

## Dataset

The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is one of the most well-known datasets in machine learning, introduced by Ronald Fisher in 1936.

| Property | Value |
|---|---|
| **Samples** | 150 (50 per class) |
| **Features** | 4 (sepal length, sepal width, petal length, petal width) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Missing Values** | None |
| **Source** | `sklearn.datasets.load_iris()` |

---

## Project Structure

```
Iris-Dataset/
├── iris_classification.py    # Main pipeline script (all 7 sections)
├── README.md                 # Project documentation
├── pairplot.png              # Pairplot of all features by species
├── heatmap.png               # Feature correlation heatmap
├── boxplot.png               # Boxplots of features grouped by species
├── scatter.png               # Petal length vs petal width scatter plot
├── knn_confusion_matrix.png  # KNN confusion matrix
├── knn_k_tuning.png          # KNN accuracy vs k (hyperparameter tuning)
├── dt_confusion_matrix.png   # Decision Tree confusion matrix
├── decision_tree.png         # Decision Tree visualization
└── model_comparison.png      # Grouped bar chart comparing both models
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MJ-Meet/Iris-Dataset.git
   cd Iris-Dataset
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

   > **Note:** Python 3.8 or higher is required.

---

## Usage

Run the complete pipeline with a single command:

```bash
python iris_classification.py
```

This will:
- Load and explore the Iris dataset
- Generate 9 publication-quality visualizations (saved as `.png`)
- Train and evaluate both KNN and Decision Tree models
- Print detailed metrics and a final comparison

---

## Pipeline Walkthrough

### 1. Data Exploration

The script loads the Iris dataset and performs exploratory data analysis:
- Dataset shape, data types, and first 10 rows
- Descriptive statistics (mean, std, quartiles)
- Class distribution (balanced: 50 samples per species)
- Null value check (confirmed: zero missing values)

### 2. Visualization

Four visualizations are generated to understand feature distributions and relationships:

| Plot | Description | File |
|---|---|---|
| **Pairplot** | Scatterplot matrix of all feature combinations, colored by species with KDE on diagonals | `pairplot.png` |
| **Heatmap** | Pearson correlation coefficients between all numeric features | `heatmap.png` |
| **Boxplot** | Distribution of each feature grouped by species (2×2 grid) | `boxplot.png` |
| **Scatter** | Petal length vs petal width — the most separable feature pair | `scatter.png` |

### 3. Preprocessing

- **Feature/Target Split:** 4 features (X) and species labels (y)
- **Train/Test Split:** 80/20 ratio with stratification (`random_state=42`)
- **Standardization:** `StandardScaler` fitted on training data only, applied to both sets
- **Result:** 120 training samples, 30 test samples

### 4. KNN Classifier

- Trained with `n_neighbors=5`, `metric='euclidean'`
- Confusion matrix saved as heatmap
- **Hyperparameter tuning:** tested k=1 through k=15
- Best k and accuracy curve plotted

### 5. Decision Tree Classifier

- Trained with `max_depth=3`, `criterion='gini'`, `random_state=42`
- Confusion matrix saved as heatmap
- Full tree structure visualized with feature names and class labels
- Feature importance ranking printed as a sorted table

### 6. Model Comparison

Side-by-side comparison of both models across four metrics:
- Accuracy, Precision (macro), Recall (macro), F1 Score (macro)
- Grouped bar chart with annotated values
- Final conclusion with recommendation

---

## Results

### Model Performance

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 Score (macro) |
|---|---|---|---|---|
| KNN (k=5) | 93.33% | 94.44% | 93.33% | 93.27% |
| **Decision Tree** | **96.67%** | **96.97%** | **96.67%** | **96.66%** |

> **Winner: Decision Tree** outperformed KNN (k=5) with 96.67% vs 93.33% accuracy on the test set.

### Feature Importances (Decision Tree)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `petal_length` | 57.91% |
| 2 | `petal_width` | 42.09% |
| 3 | `sepal_width` | 0.00% |
| 4 | `sepal_length` | 0.00% |

---

## Key Findings

- **Petal features dominate classification** — petal length and petal width alone are sufficient to separate the three species, while sepal measurements contribute zero importance in the Decision Tree.
- **Setosa is perfectly separable** — both models achieve 100% precision and recall on Setosa, which is linearly separable from the other two species.
- **Versicolor and Virginica overlap** — the slight misclassifications occur between these two species, which have overlapping feature distributions.
- **Decision Tree is preferred** for this dataset due to higher accuracy and better interpretability.
- **KNN benefits from tuning** — the best k=1 achieved 96.67% (matching Decision Tree), showing that the default k=5 was suboptimal.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Core plotting and visualization |
| **Seaborn** | Statistical data visualization |
| **scikit-learn** | ML models, preprocessing, and metrics |

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with Python and scikit-learn</i>
</p>
