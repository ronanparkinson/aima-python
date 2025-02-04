# -----------------------------
# STEP 1: SETUP AND DEBUGGING
# -----------------------------

import sys
sys.path.append('./aima-python')  # Ensure AIMA-Python package is accessible

# Import necessary modules from AIMA and standard libraries
from learning import NaiveBayesLearner, DataSet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import ListedColormap
import numpy as np

def debug_find_means_and_deviations(self):
    """
    Debug function to calculate means and standard deviations.
    Provides warnings for mismatched feature counts in the dataset.
    """
    from statistics import mean, stdev
    target_names = self.values[self.target]
    feature_numbers = len(self.inputs)

    item_buckets = self.split_values_by_classes()  # Group items by class label

    means = defaultdict(lambda: [0] * feature_numbers)
    deviations = defaultdict(lambda: [0] * feature_numbers)

    for t in target_names:
        features = [[] for _ in range(feature_numbers)]  # Initialize feature buckets
        for item in item_buckets[t]:
            if len(item) != feature_numbers:  # Warn and skip rows with mismatched feature counts
                print(f"Warning: Item {item} has {len(item)} features instead of {feature_numbers}. Skipping.")
                continue
            for i in range(feature_numbers):
                features[i].append(item[i])  # Add feature values by index

        # Calculate means and standard deviations
        for i in range(feature_numbers):
            if len(features[i]) > 1:
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])
            else:  # Handle cases with insufficient data
                means[t][i] = mean(features[i]) if features[i] else 0
                deviations[t][i] = 0

    return means, deviations

# Replace the original function with the debug version
DataSet.find_means_and_deviations = debug_find_means_and_deviations

# -----------------------------
# STEP 2: DATASET SELECTION AND EXPLORATION
# -----------------------------

print("\nDataset Selection and Preparation\n")
iris_data = load_iris()  # Load Iris dataset
X = iris_data.data  # Feature data
y = iris_data.target  # Target labels
feature_names = iris_data.feature_names  # Feature names for clarity
target_names = iris_data.target_names  # Class names

# Visualize pairwise feature relationships to explore independence assumption
print("Exploring Feature Relationships")
sns.pairplot(pd.DataFrame(X, columns=feature_names), diag_kind='kde')
plt.show()  # Displays scatterplots and KDEs for feature relationships

# -----------------------------
# STEP 3: CALCULATING PRIOR PROBABILITIES
# -----------------------------

# Split dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to Pandas DataFrame for easier manipulation
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test

# Calculate and display prior probabilities (P(C) for each class)
print("\nCalculating Prior Probabilities\n")
class_priors = train_df['target'].value_counts(normalize=True)  # Class distribution
print("Prior Probabilities (P(C)):\n", class_priors)  # Output prior probabilities

# -----------------------------
# STEP 4: IMPLEMENTING NAIVE BAYES
# -----------------------------

print("\nImplementing Naive Bayes\n")

# Convert dataset to AIMA-compatible format (list of lists)
train_data = [list(row) for row in train_df.values]
test_data = [list(row) for row in test_df.values]

# Filter rows with the correct number of features (4 features + 1 target)
expected_length = 5
train_data = [row for row in train_data if len(row) == expected_length]
test_data = [row for row in test_data if len(row) == expected_length]

# Print number of valid training and testing examples
print(f"Filtered training examples: {len(train_data)}")
print(f"Filtered testing examples: {len(test_data)}")

# Instantiate Naive Bayes Learner using the AIMA library
nb_learner = NaiveBayesLearner(DataSet(examples=train_data, target=-1))

# -----------------------------
# STEP 5: EVALUATING THE CLASSIFIER
# -----------------------------

# Predict class for test examples
predictions = [nb_learner(row[:-1]) for row in test_data]
actuals = [row[-1] for row in test_data]  # Actual class labels

# Calculate evaluation metrics (accuracy, precision, recall, etc.)
print("\nEvaluation Metrics\n")
accuracy = accuracy_score(actuals, predictions)  # Accuracy of the model
report = classification_report(actuals, predictions, target_names=target_names)  # Detailed metrics
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# -----------------------------
# STEP 6: VISUALIZING DECISION BOUNDARIES (OPTIONAL)
# -----------------------------

def plot_decision_boundaries_for_all_pairs(model, X, y, feature_names, resolution=0.01):
    """
    Plot decision boundaries for all combinations of two features.
    """
    feature_combinations = list(combinations(range(X.shape[1]), 2))  # All 2-feature pairs
    for feature1, feature2 in feature_combinations:
        # Create grid for visualization
        x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
        y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, resolution),
                               np.arange(y_min, y_max, resolution))

        # Fill grid data with predictions from the model
        grid_data = np.zeros((xx1.size, X.shape[1]))
        grid_data[:, feature1] = xx1.ravel()
        grid_data[:, feature2] = xx2.ravel()
        Z = np.array([model(row) for row in grid_data])
        Z = Z.reshape(xx1.shape)

        # Plot decision boundary
        plt.figure(figsize=(6, 6))
        plt.contourf(xx1, xx2, Z, alpha=0.8, cmap=ListedColormap(['red', 'blue', 'green']))
        plt.scatter(X[:, feature1], X[:, feature2], c=y, edgecolor='k', cmap=ListedColormap(['red', 'blue', 'green']))
        plt.xlabel(feature_names[feature1])
        plt.ylabel(feature_names[feature2])
        plt.title(f"Decision Boundary: {feature_names[feature1]} vs {feature_names[feature2]}")
        plt.show()

# Call the function to visualize decision boundaries
plot_decision_boundaries_for_all_pairs(nb_learner, X_train, y_train, feature_names)
