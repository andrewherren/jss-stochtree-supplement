##################################################################
### Setup
##################################################################

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import norm
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from stochtree import (
    BARTModel, 
    BCFModel,
    OutcomeModel,
    StochTreeBARTRegressor, 
    StochTreeBARTBinaryClassifier, 
    RNG,
    Dataset,
    Forest,
    ForestContainer,
    ForestSampler,
    GlobalVarianceModel,
    Residual,
    ForestModelConfig,
    GlobalModelConfig,
)

# Set seed for reproducibility
random_seed = 4321
rng = np.random.default_rng(random_seed)

##################################################################
### Helper Functions
##################################################################

### Data generation

def friedman_mean(x: np.array) -> np.array:
    """
    Helper function to generate "Friedman 1" dataset
    """
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * np.power(x[:, 2] - 0.5, 2.0)
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


def gaussian_error(
    rng: np.random.Generator, n: int, cond_mean: np.array = None, snr: float = None
) -> np.array:
    """
    Helper function to generate additive gaussian noise
    """
    if snr is not None and cond_mean is None:
        raise ValueError("cond_mean must be provided if snr is, and vice verse")
    if snr is not None:
        noise_sd = np.std(cond_mean) / snr
    else:
        noise_sd = 1
    return rng.normal(loc=0.0, scale=noise_sd, size=n)


def prog_fn(x: np.array) -> np.array:
    """
    Simulated prognostic / control function
    """
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * np.power(x[:, 2] - 0.5, 2.0)
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


def propensity_fn(x: np.array) -> np.array:
    """
    Simulated propensity function
    """
    return norm.cdf(0.05 * (prog_fn(x) - np.mean(prog_fn(x))))


def cate_fn(x: np.array) -> np.array:
    """
    Simulated treatment effect function
    """
    return 5 * x[:, 0]


### Data Manipulation


def compute_test_train_indices(n: int, test_set_pct: float, random_seed: int = None) -> tuple:
    """
    Helper functions for test / train split
    """
    train_inds, test_inds = train_test_split(np.arange(n), test_size=test_set_pct, random_state=random_seed)
    return (test_inds, train_inds)


def subset_data(
    data: Union[np.array, pd.DataFrame], subset_inds: np.array
) -> Union[np.array, pd.DataFrame]:
    """
    Subset the data based on user-supplied indices.
    """
    if data.ndim == 2:
        if isinstance(data, np.ndarray):
            return data[subset_inds, :]
        else:
            return data.iloc[subset_inds, :]
    else:
        return data[subset_inds]


### Custom Model Fitting Routines


def linreg_partial_residual(y: np.array, dataset: Dataset, forest: Forest) -> np.array:
    """
    Compute partial residual, net of forest predictions, as one-dimensional array
    """
    yhat_forest = forest.predict(dataset)
    return y - yhat_forest


def sample_linreg_gamma_gibbs(
    y: np.array, X: np.array, sigma2: float, tau: float, rng: np.random.Generator
) -> float:
    """
    Gibbs draw of regression coefficients, conditional on forest
    """
    y_ = y.squeeze()
    X_ = X.squeeze()
    gamma_posterior_mean = np.sum(y_ * X_) / (sigma2 + np.sum(X_ * X_))
    gamma_posterior_var = (sigma2 * tau) / (sigma2 + np.sum(X_ * X_))
    return rng.normal(
        loc=gamma_posterior_mean, scale=np.sqrt(gamma_posterior_var), size=1
    )[0]


def sample_phi_i(
    y: np.array,
    dataset: Dataset,
    forest: Forest,
    a2: float,
    tau2: float,
    nu: float,
    rng: np.random.Generator,
) -> np.array:
    """
    Sample observation-specific variance parameters phi_i
    """
    n = len(y)
    yhat_forest = forest.predict(dataset)
    res = y - yhat_forest
    posterior_shape = (nu + 1) / 2
    posterior_scale = (nu * tau2 + (res * res / a2)) / 2
    return 1 / rng.gamma(shape=posterior_shape, scale=1 / posterior_scale, size=n)


def sample_a2(
    y: np.array,
    dataset: Dataset,
    forest: Forest,
    phi_i: np.array,
    rng: np.random.Generator,
) -> float:
    """
    Sample variance parameter a^2
    """
    n = len(y)
    yhat_forest = forest.predict(dataset)
    res = y - yhat_forest
    posterior_shape = n / 2
    posterior_scale = (1 / 2) * np.sum(res * res / phi_i)
    return 1 / rng.gamma(shape=posterior_shape, scale=1 / posterior_scale, size=1)[0]


def sample_tau2(phi_i: np.array, nu: float, rng: np.random.Generator) -> float:
    """
    Sample variance parameter tau^2
    """
    n = len(phi_i)
    posterior_shape = nu * n / 2
    posterior_scale = (nu / 2) * np.sum(1 / phi_i)
    return 1 / rng.gamma(shape=posterior_shape, scale=1 / posterior_scale, size=1)[0]


##################################################################
### Section 1: Friedman Dataset Supervised Learning Demo
##################################################################

# Simulate data
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
subset_inds_list = compute_test_train_indices(n, test_set_pct, random_seed)
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
    general_params={"random_seed": random_seed, "num_threads": 1},
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
    general_params={"random_seed": random_seed, "num_threads": 1},
)

# Now inspect the resulting model
print(bart_model)

# Obtain posterior mean predictions
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
sigma2_global_samples = bart_model.extract_parameter("sigma2_global")
plt.plot(sigma2_global_samples, linestyle="-")
plt.savefig("figures/Python/friedman-bart-traceplot-warm-start.pdf")

##################################################################
### Section 2: Heteroskedasticity Demo
##################################################################

# Load the motorcycle data
mcycle_df = pd.read_csv("data/mcycle.csv")
mcycle = mcycle_df.to_numpy()
n = mcycle.shape[0]

# Run homoskedastic stochtree
num_gfr = 10
num_burnin = 0
num_mcmc = 100
general_params = {"sample_sigma2_global": True, 
                  "random_seed": random_seed, 
                  "num_threads": 1}
bart_model = BARTModel()
bart_model.sample(
    X_train=mcycle[:, 0],
    y_train=mcycle[:, 1],
    num_gfr=num_gfr,
    num_burnin=num_burnin,
    num_mcmc=num_mcmc,
    general_params=general_params,
)
sigma2_samples = bart_model.extract_parameter("sigma2_global")
sigma_samples = np.sqrt(sigma2_samples)

# Sample from the posterior predictive distribution of the homoskedastic model
y_posterior_predictive = bart_model.sample_posterior_predictive(
    X=mcycle[:, 0], num_draws_per_sample=1000
)

# Compute posterior mean
y_hat_train_hom = bart_model.predict(X=mcycle[:, 0], type="mean", terms="y_hat")

# Compute predictive intervals for the homoskedastic model
pred_interval_lb_hom = np.quantile(y_posterior_predictive, axis=(1, 2), q=0.025)
pred_interval_ub_hom = np.quantile(y_posterior_predictive, axis=(1, 2), q=0.975)

# Add a variance forest and re-run stochtree
general_params = {
    "sample_sigma2_global": False,
    "random_seed": random_seed,
    "num_threads": 1
}
variance_forest_params = {
    "num_trees": 20,
    "alpha": 0.5,
    "beta": 3.0,
    "min_samples_leaf": 20,
}
bart_model_het = BARTModel()
bart_model_het.sample(
    X_train=mcycle[:, 0],
    y_train=mcycle[:, 1],
    num_gfr=num_gfr,
    num_burnin=num_burnin,
    num_mcmc=num_mcmc,
    general_params=general_params,
    variance_forest_params=variance_forest_params,
)

# Sample from the posterior predictive distribution of the heteroskedastic model
y_posterior_predictive_het = bart_model_het.sample_posterior_predictive(
    X=mcycle[:, 0], num_draws_per_sample=10
)

# Compute posterior mean of the heteroskedastic model
y_hat_train_het = bart_model_het.predict(X=mcycle[:, 0], type="mean", terms="y_hat")

# Compute predictive intervals for the heteroskedastic model
pred_interval_lb_het = np.quantile(y_posterior_predictive_het, axis=(1, 2), q=0.025)
pred_interval_ub_het = np.quantile(y_posterior_predictive_het, axis=(1, 2), q=0.975)

# Side-by-side comparison of homoskedastic and heteroskedastic prediction intervals
plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
plt.ylim(-170, 120)
axes[0].scatter(mcycle[:, 0], mcycle[:, 1])
axes[0].plot(mcycle[:, 0], y_hat_train_hom, color="red", linewidth=2)
axes[0].plot(
    mcycle[:, 0],
    pred_interval_lb_hom,
    color="darkcyan",
    linewidth=2,
    linestyle="dashed",
)
axes[0].plot(
    mcycle[:, 0],
    pred_interval_ub_hom,
    color="darkcyan",
    linewidth=2,
    linestyle="dashed",
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Homoskedastic\nBART")
axes[1].scatter(mcycle[:, 0], mcycle[:, 1])
axes[1].plot(mcycle[:, 0], y_hat_train_het, color="red", linewidth=2)
axes[1].plot(
    mcycle[:, 0],
    pred_interval_lb_het,
    color="darkcyan",
    linewidth=2,
    linestyle="dashed",
)
axes[1].plot(
    mcycle[:, 0],
    pred_interval_ub_het,
    color="darkcyan",
    linewidth=2,
    linestyle="dashed",
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Heteroskedastic\nBART")
plt.savefig("figures/Python/motorcycle-model-comparison.pdf")

##################################################################
### Section 3: Regression Discontinuity Design BART Demo
##################################################################

# Load and organize data
data = pd.read_csv("data/CIT_2024_CUP_discrete.csv")
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
    "num_threads": 1
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

##################################################################
### Section 4: Custom Interface Additive Linear Model Demo
##################################################################

# Simulate data
n = 500
p = 10
p_W = 1
snr = 2.0
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
m_x = friedman_mean(X)
W = rng.uniform(low=0.0, high=1.0, size=(n, p_W))
gamma_W = 5.0
lm_term = (W * gamma_W).squeeze()
cond_mean = m_x + lm_term
eps = gaussian_error(rng, n, cond_mean, snr)
y = cond_mean + eps
sigma2_true = np.var(eps)

# Standardize outcome
y_bar = np.mean(y)
y_std = np.std(y)
y_standardized = (y - y_bar) / y_std

# Set model parameters
outcome_model_type = 0
leaf_dimension = 1
sigma2_init = 1.0
gamma_init = 0.0
gamma_tau = 10
num_trees = 200
feature_types = np.repeat(0, p).astype(int)  # 0 = numeric
var_weights = np.repeat(1 / p, p)

# Dataset (covariates)
forest_dataset = Dataset()
forest_dataset.add_covariates(X)

# Residual
residual = Residual(y_standardized)

# Forest wrappers
forest_container = ForestContainer(num_trees, leaf_dimension, True, False)
active_forest = Forest(num_trees, leaf_dimension, True, False)

# Model config
global_model_config = GlobalModelConfig(global_error_variance=sigma2_init)
forest_model_config = ForestModelConfig(
    feature_types=feature_types,
    num_trees=num_trees,
    num_features=p,
    num_observations=n,
    variable_weights=var_weights,
    leaf_dimension=leaf_dimension,
    leaf_model_type=outcome_model_type,
)

# Sampler and random number generator
forest_sampler = ForestSampler(forest_dataset, global_model_config, forest_model_config)
cpp_rng = RNG(random_seed)

# Variance models
global_var_model = GlobalVarianceModel()

# Initialize the leaves of each tree in the mean forest
leaf_init = np.mean(y_standardized, keepdims=True)
forest_sampler.prepare_for_sampler(
    forest_dataset,
    residual,
    active_forest,
    outcome_model_type,
    leaf_init,
)

# Prepare to run the sampler
num_burnin = 2000
num_mcmc = 2000
rmse_samples = np.empty(num_mcmc)
gamma_samples = np.empty(num_mcmc)
global_var_samples = np.empty(num_mcmc)
current_gamma = gamma_init
current_sigma2 = sigma2_init
lm_term_estimate = (W * current_gamma).squeeze()

# Run the MCMC sampler
keep_sample = False
for i in range(num_burnin + num_mcmc):
    if i >= num_burnin:
        keep_sample = True

    # Update partial residual, both in the C++ object and the R vector
    # used for coefficient sampling
    partial_res = linreg_partial_residual(y_standardized, forest_dataset, active_forest)
    residual.add_vector(lm_term_estimate)

    # Sample gamma from bayesian linear model with gaussian prior
    current_gamma = sample_linreg_gamma_gibbs(
        partial_res,
        W[:, 0],
        current_sigma2,
        gamma_tau,
        rng,
    )
    if keep_sample:
        gamma_samples[i - num_burnin] = current_gamma * y_std

    # Update partial residual before sampling forest
    lm_term_estimate = (W * current_gamma).squeeze()
    residual.subtract_vector(lm_term_estimate)

    # Sample from the forest
    forest_sampler.sample_one_iteration(
        forest_container=forest_container,
        forest=active_forest,
        dataset=forest_dataset,
        residual=residual,
        rng=cpp_rng,
        global_config=global_model_config,
        forest_config=forest_model_config,
        keep_forest=keep_sample,
        gfr=False,
        num_threads=1
    )

    # Sample global variance parameter
    current_sigma2 = global_var_model.sample_one_iteration(residual, cpp_rng, 1.0, 1.0)
    global_model_config.update_global_error_variance(current_sigma2)
    if keep_sample:
        global_var_samples[i - num_burnin] = current_sigma2 * y_std * y_std

    # Compute in-sample RMSE and cache mean function samples
    if keep_sample:
        yhat = (
            active_forest.predict(forest_dataset) * y_std
            + lm_term_estimate * y_std
            + y_bar
        )
        error = cond_mean - yhat
        rmse_samples[i - num_burnin] = np.sqrt(np.mean(error**2))


# Inspect histogram of gamma
plt.clf()
plt.hist(gamma_samples, bins=20, density=True)
plt.axvline(x=gamma_W, color="black", linestyle="--")
plt.savefig("figures/Python/custom-interface-bart-reg-gamma-histogram.pdf")

##################################################################
### Section 5: Custom Interface Robust Error Demo
##################################################################

# Simulate data
n = 1000
p = 20
X = rng.uniform(low=0.0, high=1.0, size=(n, p))
m_x = friedman_mean(X)
sigma2 = 9
nu = 2
eps = rng.standard_t(df=nu, size=n) * np.sqrt(sigma2)
y = m_x + eps
sigma2_true = np.var(eps)

# Standardize outcome
y_bar = np.mean(y)
y_std = np.std(y)
y_standardized = (y - y_bar) / y_std

# Initial values of robust model parameters
tau2_init = 1.0
a2_init = 1.0
sigma2_init = tau2_init * a2_init
phi_i_init = np.repeat(1.0, n)

# Initialize data objects
forest_dataset = Dataset()
forest_dataset.add_covariates(X)
forest_dataset.add_variance_weights(1.0 / phi_i_init)
residual = Residual(y_standardized)

# Random number generator (std::mt19937)
cpp_rng = RNG(random_seed)

# Model configuration
outcome_model_type = 0
leaf_dimension = 1
num_trees = 200
feature_types = np.repeat(0, p).astype(int)  # 0 = numeric
var_weights = np.repeat(1 / p, p)
forest_model_config = ForestModelConfig(
    feature_types=feature_types,
    num_trees=num_trees,
    num_features=p,
    num_observations=n,
    variable_weights=var_weights,
    leaf_dimension=leaf_dimension,
    leaf_model_type=outcome_model_type,
)
global_model_config = GlobalModelConfig(global_error_variance=sigma2_init)

# Forest model object
forest_sampler = ForestSampler(
    forest_dataset, 
    global_model_config, 
    forest_model_config
)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
active_forest = Forest(num_trees, leaf_dimension, True, False)
forest_container = ForestContainer(num_trees, leaf_dimension, True, False)

# Initialize the leaves of each tree in the mean forest
leaf_init = np.mean(y_standardized, keepdims=True)
forest_sampler.prepare_for_sampler(
    forest_dataset,
    residual,
    active_forest,
    outcome_model_type,
    leaf_init,
)

# Prepare to run the sampler
num_burnin = 3000
num_mcmc = 1000
sigma2_samples = np.empty(num_mcmc)
a2_samples = np.empty(num_mcmc)
tau2_samples = np.empty(num_mcmc)
phi_i_samples = np.empty((n, num_mcmc))
rmse_samples = np.empty(num_mcmc)
fhat_samples = np.empty((n, num_mcmc))
current_sigma2 = sigma2_init
current_a2 = a2_init
current_tau2 = tau2_init
current_phi_i = phi_i_init

# Run the MCMC sampler
keep_sample = False
for i in range(num_burnin + num_mcmc):
    if i >= num_burnin:
        keep_sample = True

    # Sample from the forest
    forest_sampler.sample_one_iteration(
        forest_container=forest_container,
        forest=active_forest,
        dataset=forest_dataset,
        residual=residual,
        rng=cpp_rng,
        global_config=global_model_config,
        forest_config=forest_model_config,
        keep_forest=keep_sample,
        gfr=False,
        num_threads=1
    )

    # Sample local variance parameters
    current_phi_i = sample_phi_i(
        y_standardized,
        forest_dataset,
        active_forest,
        current_a2,
        current_tau2,
        nu,
        rng,
    )

    # Sample a2
    current_a2 = sample_a2(
        y_standardized,
        forest_dataset,
        active_forest,
        current_phi_i,
        rng,
    )

    # Sample tau2
    current_tau2 = sample_tau2(current_phi_i, nu, rng)
    if keep_sample:
        tau2_samples[i - num_burnin] = current_tau2 * y_std * y_std
        sigma2_samples[i - num_burnin] = current_tau2 * current_a2 * y_std * y_std

    # Update observation-specific variance weights
    forest_dataset.update_variance_weights(current_phi_i * current_a2)

    # Compute in-sample RMSE and cache mean function samples
    if keep_sample:
        yhat_forest = active_forest.predict(forest_dataset) * y_std + y_bar
        error = m_x - yhat_forest
        rmse_samples[i - num_burnin] = np.sqrt(np.mean(error * error))
        fhat_samples[:, i - num_burnin] = yhat_forest

# Compute posterior mean of conditional expectations for the non-robust model
m_x_hat_posterior_mean = np.mean(fhat_samples, axis=1)

# Now prepare to run the same sampler without robust errors for comparison

# Initial value of global error variance parameter
sigma2_init = 1.0

# Initialize data objects
forest_dataset = Dataset()
forest_dataset.add_covariates(X)
residual = Residual(y_standardized)

# Random number generator (std::mt19937)
cpp_rng = RNG(random_seed)

# Model configuration
outcome_model_type = 0
leaf_dimension = 1
num_trees = 200
feature_types = np.repeat(0, p).astype(int)  # 0 = numeric
var_weights = np.repeat(1 / p, p)
global_model_config = GlobalModelConfig(global_error_variance=sigma2_init)
forest_model_config = ForestModelConfig(
    feature_types=feature_types,
    num_trees=num_trees,
    num_features=p,
    num_observations=n,
    variable_weights=var_weights,
    leaf_dimension=leaf_dimension,
    leaf_model_type=outcome_model_type,
)

# Forest model object
forest_sampler = ForestSampler(forest_dataset, global_model_config, forest_model_config)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
active_forest = Forest(num_trees, leaf_dimension, True, False)
forest_container = ForestContainer(num_trees, leaf_dimension, True, False)

# Initialize the leaves of each tree in the mean forest
leaf_init = np.mean(y_standardized, keepdims=True)
forest_sampler.prepare_for_sampler(
    forest_dataset,
    residual,
    active_forest,
    outcome_model_type,
    leaf_init,
)

# Global error variance model
global_var_model = GlobalVarianceModel()

# Prepare to run the sampler
num_burnin = 3000
num_mcmc = 1000
sigma2_samples_non_robust = np.empty(num_mcmc)
rmse_samples_non_robust = np.empty(num_mcmc)
fhat_samples_non_robust = np.empty((n, num_mcmc))
current_sigma2 = sigma2_init

# Run the MCMC sampler
keep_sample = False
for i in range(num_burnin + num_mcmc):
    if i >= num_burnin:
        keep_sample = True

    # Sample from the forest
    forest_sampler.sample_one_iteration(
        forest_container=forest_container,
        forest=active_forest,
        dataset=forest_dataset,
        residual=residual,
        rng=cpp_rng,
        global_config=global_model_config,
        forest_config=forest_model_config,
        keep_forest=keep_sample,
        gfr=False,
        num_threads=1
    )

    # Sample global variance parameter
    current_sigma2 = global_var_model.sample_one_iteration(residual, cpp_rng, 1.0, 1.0)
    global_model_config.update_global_error_variance(current_sigma2)
    if keep_sample:
        sigma2_samples_non_robust[i - num_burnin] = current_sigma2 * y_std * y_std

    # Compute in-sample RMSE and cache mean function samples
    if keep_sample:
        yhat_forest = active_forest.predict(forest_dataset) * y_std + y_bar
        error = m_x - yhat_forest
        rmse_samples_non_robust[i - num_burnin] = np.sqrt(np.mean(error * error))
        fhat_samples_non_robust[:, i - num_burnin] = yhat_forest

# Plot RMSE samples side-by-side
plt.clf()
y_bounds = (
    np.min([rmse_samples, rmse_samples_non_robust]) * 0.8,
    np.max([rmse_samples, rmse_samples_non_robust]) * 1.25,
)
plt.ylim(y_bounds)
plt.plot(rmse_samples, label="t Errors", color="blue")
plt.plot(
    rmse_samples_non_robust,
    label="Gaussian Errors",
    color="red",
)
plt.ylabel("In-Sample RMSE")
plt.xlabel("Iteration")
plt.legend(loc="upper left")
plt.savefig("figures/Python/custom-interface-bart-robust-rmse-comparison.pdf")

# Compute posterior mean of conditional expectations for the non-robust model
plt.clf()
m_x_hat_posterior_mean_non_robust = np.mean(fhat_samples_non_robust, axis=1)
y_bounds = (np.min(m_x) * 0.9, np.max(m_x) * 1.1)
plt.ylim(y_bounds)
plt.scatter(
    m_x_hat_posterior_mean_non_robust, m_x, label="Gaussian Errors", color="lightgray"
)
plt.scatter(m_x_hat_posterior_mean, m_x, label="t Errors", color="black")
plt.axline((np.mean(m_x), np.mean(m_x)), slope=1, color="black", linestyle=(0, (3, 3)))
plt.ylabel("True f(x)")
plt.xlabel("Predicted f(x)")
plt.legend(loc="upper left")
plt.savefig("figures/Python/custom-interface-bart-robust-pred-actual-comparison.pdf")

##################################################################
### Section 6: Simulated Causal Inference Demo
##################################################################

# Data generation parameters
n = 1000
p = 5

# Generate covariates
X = rng.uniform(0, 1, size=(n, p))

# Generate model components that are functions of X
mu_x = prog_fn(X)
pi_x = propensity_fn(X)
tau_x = cate_fn(X)

# Generate treatment
Z = rng.binomial(1, pi_x, size=n)

# Generate outcome
E_Y_ZX = mu_x + tau_x * Z
y = E_Y_ZX + rng.normal(0, 1, size=n)

# Fit a propensity model
general_params_propensity = {
  "outcome_model": OutcomeModel(outcome="binary", link="probit"),
  "sample_sigma2_global": False,
  "random_seed": random_seed,
  "num_threads": 1
}
propensity_model = BARTModel()
propensity_model.sample(
  X_train = X,
  y_train = Z,
  general_params = general_params_propensity
)

# Compute propensity scores based on the propensity model
propensity = propensity_model.predict(
  X = X,
  type = "mean",
  terms = "y_hat"
)

# Fit a causal model
bcf_model = BCFModel()
bcf_model.sample(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = propensity,
  num_gfr = 10,
  num_burnin = 2000,
  num_mcmc = 1000,
  general_params = {'random_seed' : random_seed, 
                    'num_threads': 1}
)

# Inspect the resulting model
print(bcf_model.summary())

# Compute outcome predictions
y_hat = bcf_model.predict(
  X = X,
  Z = Z,
  propensity = propensity,
  type = "mean",
  terms = "y_hat"
)

# Compute the posterior of the treatment effect function
tau_hat_posterior = bcf_model.predict(
  X = X,
  Z = Z,
  propensity = propensity,
  type = "posterior",
  terms = "cate"
)

# Plot the true CATE against the CATE posterior mean
plt.clf()
plt.scatter(np.mean(tau_hat_posterior, axis = 1), tau_x)
plt.axline((np.mean(tau_x), np.mean(tau_x)), slope = 1, color = "black", linestyle = (0, (3, 3)))
plt.xlabel("Estimated CATE")
plt.ylabel("True CATE")
plt.savefig("figures/Python/simulated-cate-true-fitted.pdf")

# ATE histogram
plt.clf()
plt.hist(np.mean(tau_hat_posterior, axis = 0), bins = 30, density = True)
plt.xlabel("ATE")
plt.axvline(x = np.mean(tau_x), color = "black", linestyle = (0, (3, 3)))
plt.savefig("figures/Python/simulated-ate-posterior.pdf")

# Fit a surrogate regression tree to the posterior mean of the CATE samples
tau_hat_bar = np.mean(tau_hat_posterior, axis=1)
surrogate_tree = DecisionTreeRegressor(random_state=random_seed, max_depth=2)
surrogate_tree.fit(X, tau_hat_bar)

# Plot the surrogate regression tree
plt.clf()
plot_tree(
    surrogate_tree,
    feature_names=[f"X{i+1}" for i in range(p)],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=8,
)
plt.savefig("figures/Python/simulated-cate-tree.pdf")

##################################################################
### Section 7: ACIC Data Causal Inference Demo
##################################################################

# Load data
df = pd.read_csv("data/acic2018_synthetic_data.csv")

# Extract treatment and outcome
y = df.loc[:, "Y"].to_numpy()
Z = df.loc[:, "Z"].to_numpy()

# Extract covariates
covariate_df = df.loc[:, ~np.isin(df.columns, ["schoolid", "Z", "Y"])]

# Encode categorical data to enable proper stochtree preprocessing
unordered_categorical_cols = ["C1", "XC"]
ordered_categorical_cols = ["S3", "C2", "C3"]
for col in unordered_categorical_cols:
    covariate_df.loc[:, col] = pd.Categorical(covariate_df.loc[:, col], ordered=False)
for col in ordered_categorical_cols:
    covariate_df.loc[:, col] = pd.Categorical(covariate_df.loc[:, col], ordered=True)

# Extract data dimensions
n, p = df.shape

# Fit a propensity model
general_params_propensity = {
    "outcome_model": OutcomeModel(outcome="binary", link="probit"),
    "sample_sigma2_global": False,
    "random_seed": random_seed,
    "num_threads": 1
}
propensity_model = BARTModel()
propensity_model.sample(
    X_train=covariate_df, y_train=Z, general_params=general_params_propensity
)
propensity = propensity_model.predict(
    X=covariate_df, type="mean", terms="y_hat"
)

# Fit a causal model
treatment_forest_params = {"keep_vars": ["X1", "X2"]}
bcf_model = BCFModel()
bcf_model.sample(
    X_train=covariate_df,
    Z_train=Z,
    y_train=y,
    propensity_train=propensity,
    num_gfr=10,
    num_burnin=2000,
    num_mcmc=1000,
    treatment_effect_forest_params=treatment_forest_params,
    general_params={"random_seed": random_seed, "num_threads": 1},
)

# Inspect the outcome predictions
y_hat = bcf_model.predict(
    X=covariate_df, Z=Z, propensity=propensity, type="mean", terms="y_hat"
)

# Retrieve posterior of treatment effect function
cate_posterior = bcf_model.predict(
    X=covariate_df, Z=Z, propensity=propensity, type="posterior", terms="cate"
)

# Compute the ATE distribution for each school
cate_posterior_df = pd.DataFrame(cate_posterior)
cate_posterior_df["schoolid"] = df.loc[:, "schoolid"].to_numpy() - 1
school_ate_posterior = cate_posterior_df.groupby("schoolid").mean().reset_index()

# Extract random effects terms
group_ids = df.loc[:, "schoolid"].to_numpy() - 1
rfx_basis = np.concatenate((np.ones((n, 1)), np.expand_dims(Z, 1)), axis=1)

# Fit causal model with random effects
bcf_model_rfx = BCFModel()
bcf_model_rfx.sample(
    X_train=covariate_df,
    Z_train=Z,
    y_train=y,
    propensity_train=propensity,
    rfx_group_ids_train=group_ids,
    rfx_basis_train=rfx_basis,
    num_gfr=10,
    num_burnin=2000,
    num_mcmc=1000,
    treatment_effect_forest_params=treatment_forest_params,
    general_params={"random_seed": random_seed, "num_threads": 1},
    random_effects_params={"model_spec": "intercept_plus_treatment"},
)

# Inspect ATE posterior by school
cate_posterior_rfx = bcf_model_rfx.predict(
    X=covariate_df,
    Z=Z,
    propensity=propensity,
    rfx_group_ids=group_ids,
    type="posterior",
    terms="cate",
)
cate_posterior_rfx_df = pd.DataFrame(cate_posterior_rfx)
cate_posterior_rfx_df["schoolid"] = group_ids.astype(int)
school_ate_rfx_posterior = cate_posterior_rfx_df.groupby("schoolid").mean().reset_index()

# Create a boxplot of random intercepts
rfx_samples = bcf_model_rfx.rfx_container.extract_parameter_samples()
rfx_betas = rfx_samples["beta_samples"]
rfx_intercept_group_means = np.array(
    [np.mean(rfx_betas[0, i, :]) for i in range(rfx_betas.shape[1])]
)
rfx_intercept_sort_inds = np.argsort(rfx_intercept_group_means).tolist()
rfx_per_group_intercept = [rfx_betas[0, i, :] for i in rfx_intercept_sort_inds]
plt.clf()
plt.boxplot(rfx_per_group_intercept)
plt.xticks([y + 1 for y in range(len(rfx_per_group_intercept)) if y % 5 == 0],
           labels=[rfx_intercept_sort_inds[y] for y in range(len(rfx_per_group_intercept)) if y % 5 == 0])
plt.xlabel('Group ID')
plt.ylabel('Random Intercept Posterior')
plt.axhline(y=0.0, color='black', linestyle='--')
plt.savefig("figures/Python/acic-random-intercept-boxplot.pdf")

# ATE Histogram
ate_posterior = np.mean(cate_posterior_rfx, axis=0)
plt.clf()
plt.hist(ate_posterior, bins = 30, density = True)
plt.savefig("figures/Python/acic-ate-posterior-rfx.pdf")

# Compare posterior of school-level ATEs for pairs of schools
# Choose the same two schools that we investigated in the R analysis
i = 48
j = 26
i_pos_rfx = school_ate_rfx_posterior.loc[school_ate_rfx_posterior.loc[:,'schoolid'] == i, 'schoolid'].iloc[0]
j_pos_rfx = school_ate_rfx_posterior.loc[school_ate_rfx_posterior.loc[:,'schoolid'] == j, 'schoolid'].iloc[0]
i_pos = school_ate_posterior.loc[school_ate_posterior.loc[:,'schoolid'] == i, 'schoolid'].iloc[0]
j_pos = school_ate_posterior.loc[school_ate_posterior.loc[:,'schoolid'] == j, 'schoolid'].iloc[0]
ate_posterior_school_i = school_ate_posterior.iloc[i_pos, 1:].to_numpy()
ate_posterior_rfx_school_i = school_ate_rfx_posterior.iloc[i_pos_rfx, 1:].to_numpy()
ate_posterior_school_j = school_ate_posterior.iloc[j_pos, 1:].to_numpy()
ate_posterior_rfx_school_j = school_ate_rfx_posterior.iloc[j_pos_rfx, 1:].to_numpy()
x_range = (np.min(np.c_[ate_posterior_school_i, ate_posterior_rfx_school_i]), 
           np.max(np.c_[ate_posterior_school_i, ate_posterior_rfx_school_i]))
y_range = (np.min(np.c_[ate_posterior_school_j, ate_posterior_rfx_school_j]), 
           np.max(np.c_[ate_posterior_school_j, ate_posterior_rfx_school_j]))
adj = 0.05
x_range = (x_range[0] - adj, x_range[1] + adj)
y_range = (y_range[0] - adj, y_range[1] + adj)
plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
axes[0].scatter(ate_posterior_school_i, ate_posterior_school_j)
axes[0].set_xlim(x_range)
axes[0].set_ylim(y_range)
axes[0].scatter(np.mean(ate_posterior_school_i), np.mean(ate_posterior_school_j), color='orange')
axes[0].axline((np.mean(ate_posterior_school_j), np.mean(ate_posterior_school_j)), slope=1, color="black", linestyle=(0, (3, 3)))
axes[0].set_xlabel(f"School {i + 1} ATE (without RFX)")
axes[0].set_ylabel(f"School {j + 1} ATE (without RFX)")
axes[1].scatter(ate_posterior_rfx_school_i, ate_posterior_rfx_school_j)
axes[1].set_xlim(x_range)
axes[1].set_ylim(y_range)
axes[1].scatter(np.mean(ate_posterior_rfx_school_i), np.mean(ate_posterior_rfx_school_j), color='orange')
axes[1].axline((np.mean(ate_posterior_rfx_school_j), np.mean(ate_posterior_rfx_school_j)), slope=1, color="black", linestyle=(0, (3, 3)))
axes[1].set_xlabel(f"School {i + 1} ATE (with RFX)")
axes[1].set_ylabel(f"School {j + 1} ATE (with RFX)")
plt.savefig("figures/Python/acic-bcf-rfx-comparison.pdf")

##################################################################
### Section 8: Sklearn API Demo
##################################################################

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

# Load binary classification dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Fit a StochTreeBARTClassifier
clf = StochTreeBARTBinaryClassifier(general_params={"random_seed": random_seed})
clf.fit(X=X, y=y)

# Predict from the model and visualize predicted probabilities
probs = clf.predict_proba(X)

# Load a multi-class dataset
dataset = load_wine()
X = dataset.data
y = dataset.target

# Fit a multi-class classification model by wrapping a OneVsRestClassifier around StochTreeBARTBinaryClassifier
clf = OneVsRestClassifier(
    StochTreeBARTBinaryClassifier(general_params={"random_seed": random_seed})
)
clf.fit(X=X, y=y)
