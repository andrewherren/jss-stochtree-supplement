# Load libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from stochtree import (
    StochTreeBARTRegressor, 
    StochTreeBARTBinaryClassifier, 
)

# Set seed for reproducibility
random_seed = 4321
rng = np.random.default_rng(random_seed)

# Simulate simple regression data
n = 100
p = 10
X = rng.normal(size=(n, p))
y = X[:, 0] * 3 + rng.normal(size=n)

# Fit a StochTreeBARTRegressor
reg = StochTreeBARTRegressor(general_params={"random_seed": random_seed})
reg.fit(X, y)

# Predict from the model and compare its (posterior mean) predictions to the true outcome
pred = reg.predict(X)
plt.clf()
plt.scatter(pred, y)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("figures/Python/sklearn-regressor-true-fitted.pdf")

# Hyperparameter tuning
param_grid = {
    "num_gfr": [10, 40],
    "num_mcmc": [0, 1000],
    "mean_forest_params": [
        {"num_trees": 50, "alpha": 0.95, "beta": 2.0},
        {"num_trees": 100, "alpha": 0.90, "beta": 1.5},
        {"num_trees": 200, "alpha": 0.85, "beta": 1.0},
    ],
}
grid_search = GridSearchCV(
    estimator=StochTreeBARTRegressor(),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
grid_search.fit(X, y)
cv_best_ind = np.argwhere(grid_search.cv_results_['rank_test_score'] == 1).item(0)
best_num_gfr = grid_search.cv_results_['param_num_gfr'][cv_best_ind].item(0)
best_num_mcmc = grid_search.cv_results_['param_num_mcmc'][cv_best_ind].item(0)
best_mean_forest_params = grid_search.cv_results_['param_mean_forest_params'][cv_best_ind]
best_num_trees = best_mean_forest_params['num_trees']
best_alpha = best_mean_forest_params['alpha']
best_beta = best_mean_forest_params['beta']
print_message = f"""
Hyperparameters chosen by grid search: 
  num_gfr: {best_num_gfr} 
  num_mcmc: {best_num_mcmc} 
  num_trees: {best_num_trees} 
  alpha: {best_alpha} 
  beta: {best_beta}
"""
print(print_message)

# Load binary classification dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Fit a StochTreeBARTClassifier
clf = StochTreeBARTBinaryClassifier(general_params={"random_seed": random_seed})
clf.fit(X=X, y=y)

# Predict from the model and visualize predicted probabilities
probs = clf.predict_proba(X)
plt.clf()
plt.hist(probs[:, 1], bins=30)
plt.xlabel("Predicted Probability")
plt.savefig("figures/Python/sklearn-classifier-binary-predicted-probabilities.pdf")

# Load a multi-class dataset
dataset = load_wine()
X = dataset.data
y = dataset.target

# Fit a multi-class classification model by wrapping a OneVsRestClassifier around StochTreeBARTBinaryClassifier
clf = OneVsRestClassifier(
    StochTreeBARTBinaryClassifier(general_params={"random_seed": random_seed})
)
clf.fit(X=X, y=y)

# Visualize the histogram of predicted probabilities for each outcome category
plt.clf()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.tight_layout(pad=3.0)
probs = clf.predict_proba(X)
ax1.hist(probs[y == 0, 0], bins=30)
ax1.set_title("Predicted Probabilities for Observations in Class 0")
ax1.set_xlim(0, 1)
ax2.hist(probs[y == 1, 1], bins=30)
ax2.set_title("Predicted Probabilities for Observations in Class 1")
ax2.set_xlim(0, 1)
ax3.hist(probs[y == 2, 2], bins=30)
ax3.set_title("Predicted Probabilities for Observations in Class 2")
ax3.set_xlim(0, 1)
plt.savefig("figures/Python/sklearn-classifier-multiclass-predicted-probabilities.pdf")
