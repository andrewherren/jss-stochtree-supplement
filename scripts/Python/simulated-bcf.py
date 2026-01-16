# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from stochtree import BARTModel, BCFModel
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Set seed for reproducibility
random_seed = 12345
rng = np.random.default_rng(random_seed)

# Simulated prognostic / control function
def prog_fn(x: np.array) -> np.array:
    return (
        10 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20 * np.power(x[:, 2] - 0.5, 2.0)
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )

# Simulated propensity function
def propensity_fn(x: np.array) -> np.array:
    return norm.cdf(0.05 * (prog_fn(x) - np.mean(prog_fn(x))))

# Simulated treatment effect function
def cate_fn(x: np.array) -> np.array:
    return 5 * x[:, 0]

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
  "probit_outcome_model": True,
  "sample_sigma2_global": False,
  "random_seed": random_seed,
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
  general_params = {'random_seed' : random_seed}
)

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
