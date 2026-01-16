# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from stochtree import (
    RNG,
    Dataset,
    Forest,
    ForestContainer,
    ForestSampler,
    GlobalVarianceModel,
    LeafVarianceModel,
    Residual,
    ForestModelConfig,
    GlobalModelConfig,
)

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


# Helper function for error distributions
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


### Helper functions for custom sampler


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


# Simulate data
rng = np.random.default_rng(random_seed)
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
        forest_container,
        active_forest,
        forest_dataset,
        residual,
        cpp_rng,
        global_model_config,
        forest_model_config,
        keep_sample,
        False,
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
