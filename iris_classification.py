"""
Iris Flower Classification Project
====================================
A complete, production-ready machine learning pipeline that classifies Iris flowers
into three species (setosa, versicolor, virginica) using their sepal and petal
measurements. This project demonstrates end-to-end ML workflow including data
exploration, visualization, preprocessing, model training (KNN & Decision Tree),
evaluation, and model comparison.

Author : Meet
Dataset: sklearn built-in Iris dataset (150 samples, 4 features, 3 classes)
"""

# ── SECTION 1: SETUP & IMPORTS ─────────────────────────────────────────────────
import pandas as pd                                  # Data manipulation and analysis
import numpy as np                                   # Numerical computing
import matplotlib.pyplot as plt                      # Core plotting library
import seaborn as sns                                # Statistical data visualization

from sklearn import datasets                         # Built-in datasets (Iris)
from sklearn.model_selection import train_test_split # Train/test splitting utility
from sklearn.preprocessing import StandardScaler     # Feature standardization (z-score)
from sklearn.neighbors import KNeighborsClassifier   # K-Nearest Neighbors classifier
from sklearn.tree import DecisionTreeClassifier      # Decision Tree classifier
from sklearn.metrics import (                        # Evaluation metrics
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.tree import plot_tree                    # Decision tree visualization

import warnings                                      # Suppress non-critical warnings
warnings.filterwarnings("ignore")


def main():
    """Main entry point that orchestrates the entire classification pipeline."""

    # ── SECTION 2: LOAD & EXPLORE DATA ──────────────────────────────────────────
    print("=" * 60)
    print("  SECTION 2: LOAD & EXPLORE DATA")
    print("=" * 60)

    # Load the classic Iris dataset bundled with scikit-learn
    iris = datasets.load_iris()

    # Build a DataFrame with readable column names instead of the default ones
    df = pd.DataFrame(
        data=iris.data,                              # 150 × 4 numpy array of measurements
        columns=["sepal_length", "sepal_width",      # Rename columns for clarity
                 "petal_length", "petal_width"],
    )

    # Append the target column (integer-coded species: 0, 1, 2)
    df["species"] = iris.target

    # Map numeric labels to human-readable species names
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = df["species"].map(species_map)   # Replace ints with strings

    # --- Exploratory summaries ---
    print(f"\nDataset shape: {df.shape}")             # Expected: (150, 5)
    print(f"\nData types:\n{df.dtypes}")              # 4 float64 + 1 object

    print(f"\nFirst 10 rows:\n{df.head(10)}")         # Quick visual sanity check

    print(f"\nDescriptive statistics:\n{df.describe()}")  # Mean, std, quartiles, etc.

    print(f"\nSpecies value counts:\n{df['species'].value_counts()}")  # Should be 50 each

    # Check for missing values — Iris is clean, but always verify
    null_counts = df.isnull().sum()
    print(f"\nNull values per column:\n{null_counts}")
    print(f"Total null values: {null_counts.sum()}")  # Expected: 0

    # ── SECTION 3: VISUALIZATION ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 3: VISUALIZATION")
    print("=" * 60)

    # --- Plot 1: Pairplot — scatterplot matrix colored by species ---
    print("\n>> Generating pairplot (all feature combinations)...")
    sns.set_style("whitegrid")                       # Clean background for all plots
    pair_grid = sns.pairplot(
        df,
        hue="species",                              # Color points by species
        palette="husl",                              # Perceptually uniform color palette
        diag_kind="kde",                             # Kernel density on the diagonal
        plot_kws={"alpha": 0.7, "s": 40},           # Semi-transparent, sized markers
    )
    pair_grid.figure.suptitle(                       # Add a main title above subplots
        "Iris Dataset — Pairplot of All Features",
        y=1.02, fontsize=14, fontweight="bold",
    )
    pair_grid.figure.tight_layout()                  # Prevent label clipping
    pair_grid.savefig("pairplot.png", dpi=150, bbox_inches="tight")  # Save to disk
    plt.close()                                      # Free memory
    print("  [OK] Saved pairplot.png")

    # --- Plot 2: Correlation heatmap with annotated coefficients ---
    print(">> Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6))           # Create figure and axes
    numeric_df = df.select_dtypes(include=[np.number])  # Only numeric columns for corr
    corr_matrix = numeric_df.corr()                  # Pearson correlation coefficients
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",                       # Show values rounded to 2 dp
        cmap="coolwarm",                             # Diverging colormap (blue-red)
        linewidths=0.5,                              # Grid lines between cells
        square=True,                                 # Square cells for readability
        vmin=-1, vmax=1,                             # Fix color scale to [-1, 1]
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved heatmap.png")

    # --- Plot 3: Boxplots — distribution of each feature grouped by species ---
    print(">> Generating boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 2×2 grid for 4 features
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for idx, (feature, ax) in enumerate(zip(features, axes.flatten())):
        sns.boxplot(
            data=df, x="species", y=feature,        # Group by species on x-axis
            palette="Set2", ax=ax,                   # Soft pastel colors
        )
        ax.set_title(f"{feature} by Species", fontsize=12, fontweight="bold")
        ax.set_xlabel("Species", fontsize=10)
        ax.set_ylabel(feature.replace("_", " ").title(), fontsize=10)  # Pretty label
    fig.suptitle("Boxplot of Features by Species", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig("boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved boxplot.png")

    # --- Plot 4: Scatter plot — petal_length vs petal_width (most separable pair) ---
    print(">> Generating scatter plot (petal_length vs petal_width)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for species_name, color in zip(species_map.values(), ["#e74c3c", "#2ecc71", "#3498db"]):
        subset = df[df["species"] == species_name]   # Filter rows for this species
        ax.scatter(
            subset["petal_length"], subset["petal_width"],
            label=species_name, alpha=0.7, s=60,     # Labeled, semi-transparent markers
            edgecolors="k", linewidth=0.5, c=color,  # Black edge for contrast
        )
    ax.set_title("Petal Length vs Petal Width by Species", fontsize=14, fontweight="bold")
    ax.set_xlabel("Petal Length (cm)", fontsize=12)
    ax.set_ylabel("Petal Width (cm)", fontsize=12)
    ax.legend(title="Species", fontsize=10, title_fontsize=11)  # Add legend with title
    fig.tight_layout()
    fig.savefig("scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved scatter.png")

    # ── SECTION 4: PREPROCESSING ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 4: PREPROCESSING")
    print("=" * 60)

    # Separate feature matrix (X) and target vector (y)
    X = df[features].values                          # Shape: (150, 4) — numpy array
    y = df["species"].values                         # Shape: (150,)   — species strings

    # Stratified split ensures each class is proportionally represented in train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,                               # 80% train, 20% test
        random_state=42,                              # Reproducible split
        stratify=y,                                   # Maintain class balance
    )

    # Standardize features: zero mean, unit variance (important for distance-based KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit on train ONLY, then transform
    X_test_scaled = scaler.transform(X_test)         # Transform test with train statistics

    print(f"\nTraining set size: {X_train_scaled.shape[0]} samples")   # Expected: 120
    print(f"Test set size:     {X_test_scaled.shape[0]} samples")     # Expected: 30

    # ── SECTION 5: MODEL 1 — K-NEAREST NEIGHBORS ───────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 5: MODEL 1 — K-NEAREST NEIGHBORS")
    print("=" * 60)

    # Train a KNN classifier with k=5 using Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train_scaled, y_train)                 # Fit on standardized training data
    y_pred_knn = knn.predict(X_test_scaled)          # Predict on unseen test data

    # Evaluate KNN performance
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"\nKNN Accuracy (k=5): {knn_accuracy:.4f}")
    print(f"\nClassification Report (KNN):\n{classification_report(y_test, y_pred_knn)}")

    # --- KNN Confusion Matrix ---
    print(">> Generating KNN confusion matrix...")
    cm_knn = confusion_matrix(y_test, y_pred_knn, labels=list(species_map.values()))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_knn,
        annot=True, fmt="d",                         # Integer annotations
        cmap="Blues",                                 # Blue gradient for KNN
        xticklabels=list(species_map.values()),       # Species names on axes
        yticklabels=list(species_map.values()),
        linewidths=1, linecolor="gray",               # Cell borders
        ax=ax,
    )
    ax.set_title("KNN Confusion Matrix (k=5)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    fig.savefig("knn_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved knn_confusion_matrix.png")

    # --- Hyperparameter tuning: find the best k from 1 to 15 ---
    print("\n>> Tuning k (1-15)...")
    k_range = range(1, 16)                           # Test k = 1 through 15
    k_accuracies = []                                # Store accuracy for each k
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_temp.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn_temp.predict(X_test_scaled))
        k_accuracies.append(acc)
        print(f"  k={k:2d}  ->  Accuracy: {acc:.4f}")

    # Identify the best k value(s) — pick the first one in case of ties
    best_k = list(k_range)[np.argmax(k_accuracies)]
    best_k_acc = max(k_accuracies)
    print(f"\n** Best k = {best_k}  with accuracy = {best_k_acc:.4f}")

    # --- Plot: Accuracy vs k ---
    print(">> Generating k-tuning plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), k_accuracies, marker="o", linestyle="-",
            color="#2980b9", linewidth=2, markersize=8, label="Test Accuracy")
    ax.axhline(y=best_k_acc, color="#e74c3c", linestyle="--",
               linewidth=1, label=f"Best Accuracy ({best_k_acc:.4f})")
    ax.axvline(x=best_k, color="#27ae60", linestyle=":",
               linewidth=1, label=f"Best k = {best_k}")
    ax.set_title("KNN — Accuracy vs. Number of Neighbors (k)", fontsize=14, fontweight="bold")
    ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(list(k_range))                     # Label every integer k
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)                               # Subtle grid lines
    fig.tight_layout()
    fig.savefig("knn_k_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved knn_k_tuning.png")

    # ── SECTION 6: MODEL 2 — DECISION TREE ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 6: MODEL 2 — DECISION TREE")
    print("=" * 60)

    # Train a Decision Tree with limited depth to prevent overfitting
    dt = DecisionTreeClassifier(
        max_depth=3,                                 # Shallow tree for interpretability
        criterion="gini",                            # Gini impurity for splitting
        random_state=42,                             # Reproducible results
    )
    dt.fit(X_train_scaled, y_train)                  # Train on standardized features
    y_pred_dt = dt.predict(X_test_scaled)            # Predict on test set

    # Evaluate Decision Tree performance
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"\nClassification Report (Decision Tree):\n{classification_report(y_test, y_pred_dt)}")

    # --- Decision Tree Confusion Matrix ---
    print(">> Generating Decision Tree confusion matrix...")
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=list(species_map.values()))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_dt,
        annot=True, fmt="d",                         # Integer annotations
        cmap="Greens",                               # Green gradient for DT
        xticklabels=list(species_map.values()),
        yticklabels=list(species_map.values()),
        linewidths=1, linecolor="gray",
        ax=ax,
    )
    ax.set_title("Decision Tree Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    fig.savefig("dt_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved dt_confusion_matrix.png")

    # --- Visualize the Decision Tree structure ---
    print(">> Generating decision tree visualization...")
    fig, ax = plt.subplots(figsize=(20, 10))         # Wide figure for tree readability
    plot_tree(
        dt,
        feature_names=features,                      # Show feature names at each node
        class_names=list(species_map.values()),       # Show species names at each leaf
        filled=True,                                 # Color nodes by majority class
        rounded=True,                                # Rounded node boxes
        fontsize=10,                                 # Readable font size
        ax=ax,
    )
    ax.set_title("Decision Tree Visualization (max_depth=3)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved decision_tree.png")

    # --- Feature importances — how much each feature contributes to splits ---
    print("\nFeature Importances (sorted, Decision Tree):")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": dt.feature_importances_,       # Gini-based importance scores
    }).sort_values(by="Importance", ascending=False) # Most important first
    importance_df.index = range(1, len(importance_df) + 1)  # 1-indexed ranking
    print(importance_df.to_string())                 # Print as a clean table

    # ── SECTION 7: MODEL COMPARISON ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SECTION 7: MODEL COMPARISON")
    print("=" * 60)

    # Extract macro-averaged precision, recall, F1 from classification reports
    knn_report = classification_report(y_test, y_pred_knn, output_dict=True)  # Dict form
    dt_report = classification_report(y_test, y_pred_dt, output_dict=True)

    # Build a comparison DataFrame with key metrics for both models
    comparison_df = pd.DataFrame({
        "Model":     ["KNN (k=5)", "Decision Tree"],
        "Accuracy":  [knn_accuracy, dt_accuracy],
        "Precision (macro)": [
            knn_report["macro avg"]["precision"],
            dt_report["macro avg"]["precision"],
        ],
        "Recall (macro)": [
            knn_report["macro avg"]["recall"],
            dt_report["macro avg"]["recall"],
        ],
        "F1 Score (macro)": [
            knn_report["macro avg"]["f1-score"],
            dt_report["macro avg"]["f1-score"],
        ],
    })

    # Display the comparison table with 4 decimal places
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False, float_format="%.4f"))

    # --- Grouped bar chart comparing all metrics side by side ---
    print("\n>> Generating model comparison chart...")
    metric_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)"]
    x = np.arange(len(metric_cols))                  # Label locations on x-axis
    bar_width = 0.3                                  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))
    # KNN bars — offset left
    bars_knn = ax.bar(
        x - bar_width / 2,
        comparison_df[metric_cols].iloc[0].values,   # KNN metric values
        bar_width, label="KNN (k=5)", color="#3498db", edgecolor="black", linewidth=0.5,
    )
    # Decision Tree bars — offset right
    bars_dt = ax.bar(
        x + bar_width / 2,
        comparison_df[metric_cols].iloc[1].values,   # DT metric values
        bar_width, label="Decision Tree", color="#2ecc71", edgecolor="black", linewidth=0.5,
    )

    # Annotate bars with their values
    for bar_group in [bars_knn, bars_dt]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",                     # 3 decimal places
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",  # Slight offset above bar
                ha="center", fontsize=9, fontweight="bold",
            )

    ax.set_title("Model Comparison — KNN vs Decision Tree", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0, 1.12)                             # Leave room for annotations
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)                     # Horizontal gridlines only
    fig.tight_layout()
    fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] Saved model_comparison.png")

    # --- Final conclusion ---
    print("\n" + "-" * 60)
    if knn_accuracy > dt_accuracy:
        winner, loser = "KNN (k=5)", "Decision Tree"
        winner_acc, loser_acc = knn_accuracy, dt_accuracy
    elif dt_accuracy > knn_accuracy:
        winner, loser = "Decision Tree", "KNN (k=5)"
        winner_acc, loser_acc = dt_accuracy, knn_accuracy
    else:
        winner = None                                # Tie scenario

    if winner:
        print(
            f"CONCLUSION: {winner} outperformed {loser} "
            f"({winner_acc:.4f} vs {loser_acc:.4f} accuracy). "
            f"{winner} is recommended for this dataset due to its superior "
            f"generalization on the test set."
        )
    else:
        print(
            f"CONCLUSION: Both models achieved identical accuracy "
            f"({knn_accuracy:.4f}). Either model is suitable; however, "
            f"Decision Tree offers better interpretability while KNN is "
            f"simpler to tune."
        )
    print("-" * 60)
    print("\nAll tasks completed successfully! Check the saved .png files.")


# ── ENTRY POINT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
