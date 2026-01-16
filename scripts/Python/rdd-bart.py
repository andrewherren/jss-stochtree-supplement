# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochtree import BARTModel
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Set seed for replicability
random_seed = 12345

# Load and organize data
url_string = (
    "https://raw.githubusercontent.com/rdpackages-replication/"
    "CIT_2024_CUP/refs/heads/main/CIT_2024_CUP_discrete.csv"
)
data = pd.read_csv(url_string)
y = data.loc[:, "nextGPA"].to_numpy().squeeze()
x = data.loc[:, "X"].to_numpy().squeeze()
n = data.shape[0]

# Standardize x
x = x / np.std(x)

# Extract covariates
w = data.iloc[:, 3:11]

# Encode categorical features as ordered/unordered factors
w["totcredits_year1"] = pd.Categorical(
    w["totcredits_year1"], ordered=True
)
unordered_categorical_cols = [
    "male",
    "bpl_north_america",
    "loc_campus1",
    "loc_campus2",
    "loc_campus3",
]
for col in unordered_categorical_cols:
    w.loc[:, col] = pd.Categorical(w.loc[:, col], ordered=False)

# x is normalized so the cutoff occurs at c = 0
c = 0

# Binarize the running variable into a "treatment" indicator
z = (x > c).astype(float)

# Window for prediction sample
h = 0.1

# Define the prediction subset
test = (-h < x) & (x < h)
ntest = np.sum(test)

# Define sampling parameters
num_gfr = 10
num_burnin = 0
num_mcmc = 500

# Define basis functions for training and testing. 
# We combine the basis for Z=1 and Z=0 to feed it to the BART call 
# and get the Y(z) predictions instantaneously. Then we separate the posterior matrix 
# between each Z and calculate the CATE prediction.
Psi = np.c_[np.ones(n), z * x, (1 - z) * x, z]

# Parameter lists for BART model fit
global_params = {
    "standardize": True,
    "sample_sigma_global": True,
    "sigma2_global_init": 0.1,
    "random_seed": random_seed,
}
forest_params = {
    "num_trees": 50,
    "min_samples_leaf": 20,
    "alpha": 0.95,
    "beta": 2,
    "max_depth": 20,
    "sample_sigma2_leaf": False,
    "sigma2_leaf_init": np.diag(np.repeat(0.1 / 50, 4)),
}

# Fit the BART model
covariates_train = w
covariates_train.loc[:, "x"] = x
bart_model = BARTModel()
bart_model.sample(
    X_train=covariates_train,
    leaf_basis_train=Psi,
    y_train=y,
    num_gfr=num_gfr,
    num_burnin=num_burnin,
    num_mcmc=num_mcmc,
    general_params=global_params,
    mean_forest_params=forest_params,
)

# Compute the CATE posterior
Psi0 = np.c_[np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)][test, :]
Psi1 = np.c_[np.ones(n), np.zeros(n), np.zeros(n), np.ones(n)][test, :]
covariates_test = w.iloc[test, :]
covariates_test.loc[:, "x"] = np.zeros(ntest)
cate_posterior = bart_model.compute_contrast(
    X_0=covariates_test,
    X_1=covariates_test,
    leaf_basis_0=Psi0,
    leaf_basis_1=Psi1,
    type="posterior",
    scale="linear",
)

# Fit regression tree
tau_hat_bar = np.mean(cate_posterior, axis=1)
surrogate_tree = DecisionTreeRegressor(random_state=random_seed, max_depth=2)
surrogate_tree.fit(w.iloc[test, :], tau_hat_bar)

# Plot regression tree
plt.clf()
plot_tree(
    surrogate_tree,
    feature_names=[
        "hsgrade_pct",
        "totcredits_year1",
        "age_at_entry",
        "male",
        "bpl_north_america",
        "loc_campus1",
        "loc_campus2",
        "loc_campus3",
    ],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8,
)
plt.savefig("figures/Python/rdd-cate-tree.pdf")
