# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from sklearn.model_selection import train_test_split
from stochtree import BARTModel

# Set seed for reproducibility
random_seed = 12345


# Helper function to generate "Friedman 1" dataset
def friedman_mean(x: np.array) -> np.array:
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * np.power(x[:, 2] - 0.5, 2.0)
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


# Helper functions for error distributions
def gaussian_error(
    rng: np.random.Generator, n: int, cond_mean: np.array = None, snr: float = None
) -> np.array:
    if snr is not None and cond_mean is None:
        raise ValueError("cond_mean must be provided if snr is, and vice verse")
    if snr is not None:
        noise_sd = np.std(cond_mean) / snr
    else:
        noise_sd = 1
    return rng.normal(loc=0.0, scale=noise_sd, size=n)


# Helper functions for test / train split
def compute_test_train_indices(n: int, test_set_pct: float) -> tuple:
    train_inds, test_inds = train_test_split(np.arange(n), test_size=test_set_pct)
    return (test_inds, train_inds)


def subset_data(
    data: Union[np.array, pd.DataFrame], subset_inds: np.array
) -> Union[np.array, pd.DataFrame]:
    if data.ndim == 2:
        if isinstance(data, np.ndarray):
            return data[subset_inds, :]
        else:
            return data.iloc[subset_inds, :]
    else:
        return data[subset_inds]


# Simulate data
rng = np.random.default_rng(random_seed)
n = 500
p = 100
snr = 3.0
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
m_x = friedman_mean(X)
eps = gaussian_error(rng, n, m_x, snr)
y = m_x + eps
sigma2_true = np.var(eps)

# Split data into test and train sets
test_set_pct = 0.2
subset_inds_list = compute_test_train_indices(n, test_set_pct)
test_inds = subset_inds_list[0]
train_inds = subset_inds_list[1]
n_test = len(test_inds)
n_train = len(train_inds)
X_test = subset_data(X, test_inds)
X_train = subset_data(X, train_inds)
y_test = subset_data(y, test_inds)
y_train = subset_data(y, train_inds)
m_x_test = subset_data(m_x, test_inds)
m_x_train = subset_data(m_x, train_inds)

# Fit a BART model from XBART initialization with different parameters for the GFR and MCMC algorithms
xbart_model = BARTModel()
xbart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=20,
    num_mcmc=0,
    general_params={"random_seed": random_seed},
)
xbart_json = xbart_model.to_json()
mean_forest_params = {"alpha": 0.25, "beta": 2, "min_samples_leaf": 10, "max_depth": 8}
bart_model = BARTModel()
bart_model.sample(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    num_gfr=0,
    num_burnin=0,
    num_mcmc=10000,
    mean_forest_params=mean_forest_params,
    previous_model_json=xbart_json,
    previous_model_warmstart_sample_num=19,
    general_params={"random_seed": random_seed},
)

# Now inspect the resulting model
y_hat_test = bart_model.predict(X=X_test, terms="y_hat", type="mean")

# Plot predicted vs true outcomes
fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=100)
axes[0].scatter(y_hat_test, y_test)
y_bar = np.mean(y_test)
axes[0].axline(
    (y_bar, y_bar), slope=1, color="black", linestyle=(0, (3, 3))
)
axes[0].set_xlabel("Predicted Conditional Mean")
axes[0].set_ylabel("Actual Outcome")
axes[1].scatter(y_hat_test, m_x_test)
mu_bar = np.mean(m_x_test)
axes[1].axline(
    (mu_bar, mu_bar), slope=1, color="black", linestyle=(0, (3, 3))
)
axes[1].set_xlabel("Predicted Conditional Mean")
axes[1].set_ylabel("Actual Conditional Mean")
plt.savefig("figures/Python/friedman-bart-pred-actual-warm-start.pdf")

# Inspect the traceplot of sigma^2
plt.clf()
plt.plot(bart_model.global_var_samples, linestyle="-")
plt.savefig("figures/Python/friedman-bart-traceplot-warm-start.pdf")
