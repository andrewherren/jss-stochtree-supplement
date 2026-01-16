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
    Residual,
    ForestModelConfig,
    GlobalModelConfig,
)

# Set seed for reproducibility
random_seed = 12345
rng = np.random.default_rng(random_seed)


# Helper function to generate "Friedman 1" dataset
def friedman_mean(x: np.array) -> np.array:
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * np.power(x[:, 2] - 0.5, 2.0)
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


## Helper functions for sampling from the robust model


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
