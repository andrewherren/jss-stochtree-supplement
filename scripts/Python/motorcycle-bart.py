# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from stochtree import BARTModel
import pyreadr

# Load the motorcycle data from its Github CRAN source mirror
url = "https://github.com/cran/MASS/blob/master/data/mcycle.rda?raw=true"
dst_path = "mcycle.rda"
dst_path_again = pyreadr.download_file(url, dst_path)
res = pyreadr.read_r(dst_path)
mcycle_df = res['mcycle']
mcycle = mcycle_df.to_numpy()
n = mcycle.shape[0]

# Seed for reproducibility
random_seed = 12345

# Run homoskedastic stochtree
num_gfr = 10
num_burnin = 0
num_mcmc = 100
general_params = {"sample_sigma2_global": True, "random_seed": random_seed}
bart_model = BARTModel()
bart_model.sample(
    X_train=mcycle[:, 0],
    y_train=mcycle[:, 1],
    num_gfr=num_gfr,
    num_burnin=num_burnin,
    num_mcmc=num_mcmc,
    general_params=general_params,
)
sigma_samples = np.sqrt(bart_model.global_var_samples)

# Sample from the posterior predictive distribution of the homoskedastic model
y_posterior_predictive = bart_model.sample_posterior_predictive(
    X=mcycle[:, 0], num_draws_per_sample=1000
)

# Compute posterior mean
y_hat_train_hom = bart_model.predict(X=mcycle[:, 0], type="mean", terms="y_hat")

# Compute predictive intervals for the homoskedastic model
pred_interval_lb_hom = np.quantile(y_posterior_predictive, axis=(0, 2), q=0.025)
pred_interval_ub_hom = np.quantile(y_posterior_predictive, axis=(0, 2), q=0.975)

# Add a variance forest and re-run stochtree
general_params = {
    "sample_sigma2_global": False,
    "random_seed": random_seed,
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
pred_interval_lb_het = np.quantile(y_posterior_predictive_het, axis=(0, 2), q=0.025)
pred_interval_ub_het = np.quantile(y_posterior_predictive_het, axis=(0, 2), q=0.975)

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
