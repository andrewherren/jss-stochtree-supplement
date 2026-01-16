# Load libraries
library(stochtree)

# Seed for reproducibility
random_seed <- 12345
set.seed(random_seed)

# Load the ACIC data
url_string <- paste0(
    "https://raw.githubusercontent.com/andrewherren/acic2024/",
    "refs/heads/main/data/acic2018/synthetic_data.csv"
)
df <- read.csv(url_string)

# Extract treatment and outcome
y <- df$Y
Z <- df$Z

# Extract covariates
covariate_df <- df[, !(colnames(df) %in% c("schoolid", "Z", "Y"))]

# Encode categorical data to enable proper stochtree preprocessing
unordered_categorical_cols <- c("C1", "XC")
ordered_categorical_cols <- c("S3", "C2", "C3")
for (col in unordered_categorical_cols) {
    covariate_df[, col] <- factor(covariate_df[, col], ordered = F)
}
for (col in ordered_categorical_cols) {
    covariate_df[, col] <- factor(covariate_df[, col], ordered = T)
}

# Extract data dimensions
n <- nrow(df)
p <- ncol(df)

# Fit a propensity model
general_params_propensity = list(
    probit_outcome_model = T,
    sample_sigma2_global = F,
    random_seed = random_seed
)
propensity_model <- stochtree::bart(
    X_train = covariate_df,
    y_train = Z,
    general_params = general_params_propensity
)
propensity <- predict(
    propensity_model,
    X = covariate_df,
    type = "mean",
    terms = "y_hat"
)

# Fit a causal model
treatment_forest_params <- list(
    keep_vars = c("X1", "X2")
)
bcf_model <- stochtree::bcf(
    X_train = covariate_df,
    Z_train = Z,
    y_train = y,
    propensity_train = propensity,
    num_gfr = 10,
    num_burnin = 2000,
    num_mcmc = 1000,
    general_params = list(random_seed = random_seed),
    treatment_effect_forest_params = treatment_forest_params
)

# Retrieve posterior of treatment effect function
cate_posterior <- predict(
    bcf_model,
    X = covariate_df,
    Z = Z,
    propensity = propensity,
    type = "posterior",
    terms = "cate"
)

# Compute the ATE distribution for each school
num_samples <- ncol(cate_posterior)
cate_posterior_df <- as.data.frame(cate_posterior)
cate_posterior_df$schoolid <- factor(df$schoolid)
school_ate_posterior <- aggregate(
    . ~ schoolid,
    data = cate_posterior_df,
    FUN = mean
)
sort_inds <- order(sapply(1:76, function(x) {
    mean(as.numeric(school_ate_posterior[
        school_ate_posterior$schoolid == x,
        2:(num_samples + 1)
    ]))
}))
school_ate_posterior <- school_ate_posterior[sort_inds, ]
school_ate_posterior$schoolid <- factor(
    sort_inds,
    levels = sort_inds,
    labels = sort_inds
)

# Extract random effects terms
group_ids <- as.integer(df$schoolid)
rfx_basis <- cbind(1, Z)

# Fit causal model with random effects
rfx_params <- list(model_spec = "intercept_plus_treatment")
bcf_model_rfx <- stochtree::bcf(
    X_train = covariate_df,
    Z_train = Z,
    y_train = y,
    propensity_train = propensity,
    rfx_group_ids_train = group_ids,
    num_gfr = 10,
    num_burnin = 2000,
    num_mcmc = 1000,
    general_params = list(random_seed = random_seed),
    treatment_effect_forest_params = treatment_forest_params,
    random_effects_params = rfx_params
)

# Inspect ATE posterior by school
cate_posterior_rfx <- predict(
    bcf_model_rfx,
    X = covariate_df,
    Z = Z,
    propensity = propensity,
    rfx_group_ids = group_ids,
    type = "posterior",
    terms = "cate"
)
num_samples_rfx <- ncol(cate_posterior_rfx)
cate_posterior_rfx_df <- as.data.frame(cate_posterior_rfx)
cate_posterior_rfx_df$schoolid <- factor(df$schoolid)
school_ate_rfx_posterior <- aggregate(
    . ~ schoolid,
    data = cate_posterior_rfx_df,
    FUN = mean
)
sort_inds <- order(sapply(1:76, function(x) {
    mean(as.numeric(school_ate_rfx_posterior[
        school_ate_rfx_posterior$schoolid == x,
        2:(num_samples_rfx + 1)
    ]))
}))
school_ate_rfx_posterior <- school_ate_rfx_posterior[sort_inds, ]
school_ate_rfx_posterior$schoolid <- factor(
    sort_inds,
    levels = sort_inds,
    labels = sort_inds
)

# Boxplots of random intercepts
rfx_samples <- getRandomEffectSamples(bcf_model_rfx)
rfx_betas <- rfx_samples$beta_samples
random_intercepts <- as.data.frame(rfx_betas[1, , ])
sort_inds <- order(sapply(1:76, function(x) {
    mean(as.numeric(random_intercepts[
        x,
        1:bcf_model_rfx$model_params$num_samples
    ]))
}))
random_intercepts <- random_intercepts[sort_inds, ]
random_intercepts$schoolid <- factor(
    sort_inds,
    levels = sort_inds,
    labels = sort_inds
)
random_intercepts_long <- reshape(
    random_intercepts,
    idvar = "schoolid",
    varying = list(1:bcf_model_rfx$model_params$num_samples),
    v.names = "V",
    direction = "long"
)
pdf("figures/R/acic-random-intercept-boxplot.pdf", width = 4, height = 3, pointsize = 8)
box_out <- boxplot(
    V ~ schoolid,
    data = random_intercepts_long,
    coef = 0,
    xlab = "School ID",
    ylab = "Intercept",
    main = "Random Intercept Posterior"
)
abline(h = 0, lty = 2, lwd = 3, col = "blue")
dev.off()

# ATE histogram
ate_posterior_rfx <- colMeans(cate_posterior_rfx)
pdf(
    "figures/R/acic-ate-posterior-rfx.pdf",
    width = 4,
    height = 3,
    pointsize = 8
)
hist(
    ate_posterior_rfx,
    xlab = "ATE",
    main = NULL,
    breaks = 30
)
dev.off()

# Compare posterior of school-level ATEs for pairs of schools
ind <- 1
i <- school_ate_posterior$schoolid[ind]
j <- school_ate_posterior$schoolid[ind + 1]
ate_posterior_school_i <- as.numeric(school_ate_posterior[
    school_ate_posterior$schoolid == i,
    2:(num_samples + 1)
])
ate_posterior_rfx_school_i <- as.numeric(school_ate_rfx_posterior[
    school_ate_rfx_posterior$schoolid == i,
    2:(num_samples + 1)
])
ate_posterior_school_j <- as.numeric(school_ate_posterior[
    school_ate_posterior$schoolid == j,
    2:(num_samples + 1)
])
ate_posterior_rfx_school_j <- as.numeric(school_ate_rfx_posterior[
    school_ate_rfx_posterior$schoolid == j,
    2:(num_samples + 1)
])
pdf(
    "figures/R/acic-bcf-rfx-comparison.pdf",
    width = 4,
    height = 3,
    pointsize = 8
)
par(mfrow = (c(1, 2)))
x_range <- range(c(ate_posterior_school_i, ate_posterior_rfx_school_i))
y_range <- range(c(ate_posterior_school_j, ate_posterior_rfx_school_j))
plot(
    ate_posterior_school_i,
    ate_posterior_school_j,
    xlab = paste0("School ", i, " ATE (without RFX)"),
    ylab = paste0("School ", j, " ATE (without RFX)"),
    xlim = x_range,
    ylim = y_range
)
abline(0, 1)
points(
    mean(ate_posterior_school_i),
    mean(ate_posterior_school_j),
    col = "orange",
    pch = 16,
    cex = 1.5
)
plot(
    ate_posterior_rfx_school_i,
    ate_posterior_rfx_school_j,
    xlab = paste0("School ", i, " ATE (with RFX)"),
    ylab = paste0("School ", j, " ATE (with RFX)"), 
    xlim = x_range,
    ylim = y_range
)
abline(0, 1)
points(
    mean(ate_posterior_rfx_school_i),
    mean(ate_posterior_rfx_school_j),
    col = "orange",
    pch = 16,
    cex = 1.5
)
dev.off()
par(mfrow = (c(1, 1)))
