# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochtree import BARTModel, BCFModel

# Set seed for reproducibility
random_seed = 12345

# Load data
url_string = "https://raw.githubusercontent.com/andrewherren/acic2024/" \
             "refs/heads/main/data/acic2018/synthetic_data.csv"
df = pd.read_csv(url_string)

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
    "probit_outcome_model": True,
    "sample_sigma2_global": False,
    "random_seed": random_seed,
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
    general_params={"random_seed": random_seed},
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
    general_params={"random_seed": random_seed},
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
