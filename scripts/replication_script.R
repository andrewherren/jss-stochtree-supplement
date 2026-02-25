##################################################################
### Setup
##################################################################

# Load packages
library(stochtree)
library(MASS)
library(rpart)
library(rpart.plot)

# Seed for reproducibility
random_seed <- 4321
set.seed(random_seed)

##################################################################
### Helper Functions
##################################################################

### Data generation

# Helper function to generate "Friedman 1" dataset
friedman_mean <- function(x) {
    10 *
        sin(pi * x[, 1] * x[, 2]) +
        20 * (x[, 3] - 0.5)^2 +
        10 * x[, 4] +
        5 * x[, 5]
}

# Helper functions for error distributions
gaussian_error <- function(n, cond_mean = NULL, snr = NULL) {
    if (!is.null(snr)) {
        stopifnot(!is.null(cond_mean))
    }
    if (!is.null(cond_mean)) {
        stopifnot(!is.null(snr))
    }
    if (!is.null(snr)) {
        noise_sd <- sd(cond_mean) / snr
    } else {
        noise_sd <- 1
    }
    return(rnorm(n, 0, 1) * noise_sd)
}

# Simulated prognostic / control function
prog_fn <- function(x) {
  10 *
    sin(pi * x[, 1] * x[, 2]) +
    20 * (x[, 3] - 0.5)^2 +
    10 * x[, 4] +
    5 * x[, 5]
}

# Simulated propensity function
propensity_fn <- function(x) {
  pnorm(0.05 * (prog_fn(x) - mean(prog_fn(x))))
}

# Simulated treatment effect function
cate_fn <- function(x) {
  5 * x[, 1]
}

### Data Manipulation

# Helper functions for test / train split
compute_test_train_indices <- function(n, test_set_pct) {
    n_test <- round(test_set_pct * n)
    n_train <- n - n_test
    test_inds <- sort(sample(1:n, n_test, replace = FALSE))
    train_inds <- (1:n)[!((1:n) %in% test_inds)]
    return(list(test_inds = test_inds, train_inds = train_inds))
}
subset_data <- function(data, subset_inds) {
    if (is.matrix(data)) {
        return(data[subset_inds, ])
    } else {
        return(data[subset_inds])
    }
}

### Custom Model Fitting Routines

# Compute partial residual, net of forest predictions, as R vector
linreg_partial_residual <- function(y, dataset, forest) {
    yhat_forest <- forest$predict(dataset)
    return(y - yhat_forest)
}

# Gibbs draw of regression coefficients, conditional on forest
sample_linreg_gamma_gibbs <- function(y, X, sigma2, tau) {
    gamma_posterior_mean <- sum(y * X) / (sigma2 + sum(X * X))
    gamma_posterior_var <- (sigma2 * tau) / (sigma2 + sum(X * X))
    return(rnorm(1, gamma_posterior_mean, sqrt(gamma_posterior_var)))
}

# Sample observation-specific variance parameters phi_i
sample_phi_i <- function(y, dataset, forest, a2, tau2, nu) {
    n <- length(y)
    yhat_forest <- forest$predict(dataset)
    res <- y - yhat_forest
    posterior_shape <- (nu + 1) / 2
    posterior_scale <- (nu * tau2 + (res * res / a2)) / 2
    return(1 / rgamma(n, posterior_shape, rate = posterior_scale))
}

# Sample variance parameter a^2
sample_a2 <- function(y, dataset, forest, phi_i) {
    n <- length(y)
    yhat_forest <- forest$predict(dataset)
    res <- y - yhat_forest
    posterior_shape <- n / 2
    posterior_scale <- (1 / 2) * sum(res * res / phi_i)
    return(1 / rgamma(1, posterior_shape, rate = posterior_scale))
}

# Sample variance parameter tau^2
sample_tau2 <- function(phi_i, nu) {
    n <- length(phi_i)
    posterior_shape <- nu * n / 2
    posterior_scale <- (nu / 2) * sum(1 / phi_i)
    return(1 / rgamma(1, posterior_shape, rate = posterior_scale))
}

##################################################################
### Section 1: Friedman Dataset Supervised Learning Demo
##################################################################

# Simulate data
n <- 500
p <- 100
snr <- 3
X <- matrix(runif(n * p), ncol = p)
m_x <- friedman_mean(X)
eps <- gaussian_error(n, m_x, snr)
y <- m_x + eps
sigma2_true <- var(eps)

# Split data into test and train sets
test_set_pct <- 0.2
subset_inds_list <- compute_test_train_indices(n, test_set_pct)
test_inds <- subset_inds_list$test_inds
train_inds <- subset_inds_list$train_inds
n_test <- length(test_inds)
n_train <- length(train_inds)
X_test <- subset_data(X, test_inds)
X_train <- subset_data(X, train_inds)
y_test <- subset_data(y, test_inds)
y_train <- subset_data(y, train_inds)
m_x_test <- subset_data(m_x, test_inds)
m_x_train <- subset_data(m_x, train_inds)

# Fit a BART model from XBART initialization with different parameters for the GFR and MCMC algorithms
xbart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 20,
    num_mcmc = 0,
    general_params = list(random_seed = random_seed, 
                          num_threads = 1)
)
xbart_json <- saveBARTModelToJsonString(xbart_model)
mean_forest_params = list(
    alpha = 0.25,
    beta = 2,
    min_samples_leaf = 10,
    max_depth = 8
)
bart_model <- bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = 10000,
    mean_forest_params = mean_forest_params,
    previous_model_json = xbart_json,
    previous_model_warmstart_sample_num = 20,
    general_params = list(random_seed = random_seed, 
                          num_threads = 1)
)

# Now inspect the resulting model
y_hat_test <- predict(bart_model, X = X_test, terms = "y_hat", type = "mean")

# Chart for paper
pdf(
    "figures/R/friedman-bart-traceplot-warm-start.pdf",
    width = 4,
    height = 3,
    pointsize = 10
)
sigma2_global_samples <- extract_parameter(bart_model, "sigma2_global")
plot(
    sigma2_global_samples,
    type = "l",
    xlab = "Iteration",
    ylab = "Error Variance Value"
)
dev.off()

# Chart for paper
pdf(
    "figures/R/friedman-bart-pred-actual-warm-start.pdf",
    width = 4,
    height = 3,
    pointsize = 10
)
par(mfrow = c(1, 2))
plot(
    y_hat_test,
    y_test,
    xlab = "Predicted Conditional\nMean",
    ylab = "Actual Outcome"
)
abline(0, 1, col = "blue", lty = 3, lwd = 3)
plot(
    y_hat_test,
    m_x_test,
    xlab = "Predicted Conditional\nMean",
    ylab = "Actual Conditional Mean"
)
abline(0, 1, col = "blue", lty = 3, lwd = 3)
dev.off()

##################################################################
### Section 2: Heteroskedasticity Demo
##################################################################

# Load the motorcycle data
mcycle <- MASS::mcycle
n <- nrow(mcycle)

# Run homoskedastic stochtree
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_samples <- num_gfr + num_burnin + num_mcmc
general_params <- list(sample_sigma2_global = T, 
                       random_seed = random_seed,
                       num_threads = 1
)
bart_model <- bart(
    X_train = as.matrix(mcycle$times),
    y_train = mcycle$accel,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params
)
sigma_samples <- sqrt(extract_parameter(bart_model, "sigma2_global"))

# Sample from the posterior predictive distribution of the homoskedastic model
y_posterior_predictive <- sample_bart_posterior_predictive(
    bart_model,
    X = as.matrix(mcycle$times),
    num_draws_per_sample = 1000
)

# Compute posterior mean of the heteroskedastic model
y_hat_train_hom <- predict(
    bart_model,
    X = as.matrix(mcycle$times),
    type = "mean",
    terms = "y_hat"
)

# Compute predictive intervals for the heteroskedastic model
pred_interval_lb_hom <- apply(
    y_posterior_predictive,
    1,
    quantile,
    probs = 0.025
)
pred_interval_ub_hom <- apply(
    y_posterior_predictive,
    1,
    quantile,
    probs = 0.975
)

# Add a variance forest and re-run stochtree
general_params <- list(sample_sigma2_global = F, random_seed = random_seed, num_threads = 1)
variance_forest_params <- list(
    num_trees = 20,
    alpha = 0.5,
    beta = 3.0,
    min_samples_leaf = 20
)
bart_model_het <- bart(
    X_train = as.matrix(mcycle$times),
    y_train = mcycle$accel,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params,
    variance_forest_params = variance_forest_params
)

# Sample from the posterior predictive distribution of the heteroskedastic model
y_posterior_predictive_het <- sample_bart_posterior_predictive(
    bart_model_het,
    X = as.matrix(mcycle$times),
    num_draws_per_sample = 10
)

# Compute posterior mean of the heteroskedastic model
y_hat_train_het <- predict(
    bart_model_het,
    X = as.matrix(mcycle$times),
    type = "mean",
    terms = "y_hat"
)

# Compute predictive intervals for the heteroskedastic model
pred_interval_lb_het <- apply(
    y_posterior_predictive_het,
    1,
    quantile,
    probs = 0.025
)
pred_interval_ub_het <- apply(
    y_posterior_predictive_het,
    1,
    quantile,
    probs = 0.975
)

# Side-by-side comparison of homoskedastic and heteroskedastic prediction intervals
pdf(
    file = "figures/R/motorcycle-model-comparison.pdf",
    width = 10,
    height = 6,
    pointsize = 22
)
par(mfrow = c(1, 2))
plot(
    mcycle$times,
    mcycle$accel,
    pch = 16,
    cex = 0.75,
    xlab = "x",
    ylab = "y",
    main = "Homoskedastic\nBART",
    ylim = c(-170, 120)
)
lines(mcycle$times, y_hat_train_hom, col = 'red', lwd = 2)
lines(mcycle$times, pred_interval_lb_hom, col = 'darkcyan', lwd = 2, lty = 2)
lines(mcycle$times, pred_interval_ub_hom, col = 'darkcyan', lwd = 2, lty = 2)
plot(
    mcycle$times,
    mcycle$accel,
    pch = 16,
    cex = 0.75,
    xlab = "x",
    ylab = "y",
    main = "Heteroskedastic\nBART",
    ylim = c(-170, 120)
)
lines(mcycle$times, y_hat_train_het, col = 'red', lwd = 2)
lines(mcycle$times, pred_interval_lb_het, col = 'darkcyan', lwd = 2, lty = 2)
lines(mcycle$times, pred_interval_ub_het, col = 'darkcyan', lwd = 2, lty = 2)
par(mfrow = c(1, 1))
dev.off()

##################################################################
### Section 3: Regression Discontinuity Design BART Demo
##################################################################

# Load and organize data
data <- read.csv(
  paste0(
    "https://raw.githubusercontent.com/rdpackages-replication/",
    "CIT_2024_CUP/refs/heads/main/CIT_2024_CUP_discrete.csv"
  )
)
y <- data$nextGPA
x <- data$X
n <- nrow(data)

# Standardize x
x <- x / sd(x) ## we always standardize X

# Extract covariates
w <- data[, 4:11]

# Encode categorical features as ordered/unordered factors
w$totcredits_year1 <- factor(w$totcredits_year1, ordered = TRUE)
w$male <- factor(w$male, ordered = FALSE)
w$bpl_north_america <- factor(w$bpl_north_america, ordered = FALSE)
w$loc_campus1 <- factor(w$loc_campus1, ordered = FALSE)
w$loc_campus2 <- factor(w$loc_campus2, ordered = FALSE)
w$loc_campus3 <- factor(w$loc_campus3, ordered = FALSE)

# x is normalized so the cutoff occurs at c = 0
c <- 0

# Binarize the running variable into a "treatment" indicator
z <- as.numeric(x > c)

# Window for prediction sample
h <- 0.1

# Define the prediction subset
test <- -h < x & x < h
ntest <- sum(test)

# Define sampling parameters
num_gfr <- 2
num_burnin <- 0
num_mcmc <- 500

# Define basis functions for training and testing.
# We combine the basis for Z=1 and Z=0 to feed it to the BART call 
# and get the Y(z) predictions instantaneously. Then we separate the posterior matrix 
# between each Z and calculate the CATE prediction.
Psi <- cbind(rep(1, n), z * x, (1 - z) * x, z)

# Parameter lists for BART model fit
global_params <- list(
  standardize = T,
  sample_sigma_global = TRUE,
  sigma2_global_init = 0.1, 
  random_seed = random_seed, 
  num_threads = 1
)
forest_params <- list(
  num_trees = 50,
  min_samples_leaf = 20,
  alpha = 0.95,
  beta = 2,
  max_depth = 20,
  sample_sigma2_leaf = FALSE,
  sigma2_leaf_init = diag(rep(0.1 / 50, 4))
)

# Fit the BART model
bart_model <- bart(
  X_train = cbind(x, w),
  leaf_basis_train = Psi,
  y_train = y,
  num_gfr = num_gfr,
  num_burnin = num_burnin,
  num_mcmc = num_mcmc,
  general_params = global_params,
  mean_forest_params = forest_params
)

# Compute the CATE posterior
Psi0 <- cbind(rep(1, n), rep(0, n), rep(0, n), rep(0, n))[test, ]
Psi1 <- cbind(rep(1, n), rep(0, n), rep(0, n), rep(1, n))[test, ]
covariates_test <- cbind(x = rep(0, n), w)[test, ]
cate_posterior <- compute_contrast_bart_model(
  bart_model,
  X_0 = covariates_test,
  X_1 = covariates_test,
  leaf_basis_0 = Psi0,
  leaf_basis_1 = Psi1,
  type = "posterior",
  scale = "linear"
)

# Fit a surrogate regression tree
summary_df <- data.frame(y = rowMeans(cate_posterior), w[test, ])
cate <- rpart(y ~ ., summary_df, control = rpart.control(cp = 0.015))
cate <- prune.rpart(cate, cp = 0.05)

# Define separate colors for left and rightmost nodes
plot.cart <- function(rpart.obj) {
  rpart.frame <- rpart.obj$frame
  left <- which.min(rpart.frame$yval)
  right <- which.max(rpart.frame$yval)
  nodes <- rep(NA, nrow(rpart.frame))
  for (i in 1:length(nodes)) {
    if (rpart.frame$yval[i] == rpart.frame$yval[right]) {
      nodes[i] <- "gold2"
    } else if (rpart.frame$yval[i] == rpart.frame$yval[left]) {
      nodes[i] <- "tomato3"
    } else {
      nodes[i] <- "lightblue3"
    }
  }
  return(nodes)
}

# Plot the surrogate regression tree
pdf("figures/R/rdd-cate-tree.pdf", width = 3.5, height = 3, pointsize = 10)
rpart.plot(cate, main = "", box.col = plot.cart(cate))
dev.off()

##################################################################
### Section 4: Custom Interface Additive Linear Model Demo
##################################################################

# Simulate data with homoscedastic Gaussian errors
n <- 500
p <- 10
p_W <- 1
snr <- 2
X <- matrix(runif(n * p), ncol = p)
m_x <- friedman_mean(X)
W <- matrix(runif(n * p_W), ncol = p_W)
gamma_W <- c(5)
lm_term <- W %*% gamma_W
cond_mean <- lm_term + m_x
eps <- gaussian_error(n, cond_mean, snr)
y <- cond_mean + eps
sigma2_true <- var(eps)
y_bar <- mean(y)
y_std <- sd(y)
y_standardized <- (y - y_bar) / y_std

# Data
forest_dataset <- createForestDataset(X)
outcome <- createOutcome(y_standardized)

# Random number generator (std::mt19937)
rng <- createCppRNG(random_seed)

# Model configuration
outcome_model_type <- 0
leaf_dimension <- 1
sigma2_init <- 1.
gamma_init <- 0.
gamma_tau <- 10
num_trees <- 200
feature_types <- as.integer(rep(0, p)) # 0 = numeric
variable_weights <- rep(1 / p, p)
forest_model_config <- createForestModelConfig(
    feature_types = feature_types,
    num_trees = num_trees,
    num_features = p,
    num_observations = n,
    variable_weights = variable_weights,
    leaf_dimension = leaf_dimension,
    leaf_model_type = outcome_model_type
)
global_model_config <- createGlobalModelConfig(
    global_error_variance = sigma2_init
)

# Forest model object
forest_model <- createForestModel(
    forest_dataset,
    forest_model_config,
    global_model_config
)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
forest_samples <- createForestSamples(num_trees, 1, T)
active_forest <- createForest(num_trees, 1, T)

# Initialize the leaves of each tree in the forest
leaf_init <- mean(y_standardized)
active_forest$prepare_for_sampler(
    forest_dataset,
    outcome,
    forest_model,
    outcome_model_type,
    leaf_init
)

# Prepare to run the MCMC sampler
num_burnin <- 2000
num_mcmc <- 2000
global_var_samples <- rep(0, num_mcmc)
gamma_samples <- rep(0, num_mcmc)
rmse_samples <- rep(0, num_mcmc)
current_gamma <- gamma_init
current_sigma2 <- sigma2_init
lm_term_estimate <- W %*% current_gamma

# Run the MCMC sampler
keep_sample <- F
for (i in 1:(num_burnin + num_mcmc)) {
    if (i > num_burnin) {
        keep_sample <- T
    }
    # Update partial residual, both in the C++ object and the R vector
    # used for coefficient sampling
    partial_res <- linreg_partial_residual(
        y_standardized,
        forest_dataset,
        active_forest
    )
    outcome$add_vector(lm_term_estimate)

    # Sample gamma from bayesian linear model with gaussian prior
    current_gamma <- sample_linreg_gamma_gibbs(
        partial_res,
        W[, 1],
        current_sigma2,
        gamma_tau
    )
    if (keep_sample) {
        gamma_samples[i - num_burnin] <- current_gamma * y_std
    }

    # Update partial residual before sampling forest
    lm_term_estimate <- W %*% current_gamma
    outcome$subtract_vector(lm_term_estimate)

    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset,
        outcome,
        forest_samples,
        active_forest,
        rng,
        forest_model_config,
        global_model_config,
        keep_forest = keep_sample,
        gfr = F, 
        num_threads = 1
    )

    # Sample global variance parameter
    current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
        outcome,
        forest_dataset,
        rng,
        1,
        1
    )
    global_model_config$update_global_error_variance(current_sigma2)
    if (keep_sample) {
        global_var_samples[i - num_burnin] <- current_sigma2 * y_std^2
    }

    # Compute in-sample RMSE and cache mean function samples
    if (keep_sample) {
        yhat <- (active_forest$predict(forest_dataset) + lm_term_estimate) * y_std + y_bar
        error <- (cond_mean - yhat)
        rmse_samples[i - num_burnin] <- sqrt(mean(error * error))
    }
}

# Look at traceplot of regression parameter samples
pdf(
    "figures/R/custom-interface-bart-reg-gamma-histogram.pdf",
    width = 4,
    height = 3,
    pointsize = 10
)
hist(
    gamma_samples,
    main = NULL,
    xlab = expression(gamma),
    breaks = 20,
    probability = F
)
abline(v = gamma_W, col = "blue", lty = 1, lwd = 4)
dev.off()

##################################################################
### Section 5: Custom Interface Robust Error Demo
##################################################################

# Simulate data
n <- 1000
p <- 20
X <- matrix(runif(n * p), ncol = p)
m_x <- friedman_mean(X)
sigma2 <- 9
nu <- 2
eps <- rt(n, df = nu) * sqrt(sigma2)
y <- m_x + eps
sigma2_true <- var(eps)
y_bar <- mean(y)
y_std <- sd(y)
y_standardized <- (y - y_bar) / y_std

# Initial values of robust model parameters
tau2_init <- 1.
a2_init <- 1.
sigma2_init <- 1.
phi_i_init <- rep(1., n)

# Initialize data objects
forest_dataset <- createForestDataset(X, variance_weights = 1 / phi_i_init)
outcome <- createOutcome(y_standardized)

# Random number generator (std::mt19937)
rng <- createCppRNG(random_seed)

# Model configuration
outcome_model_type <- 0
leaf_dimension <- 1
num_trees <- 200
feature_types <- as.integer(rep(0, p)) # 0 = numeric
variable_weights <- rep(1 / p, p)
forest_model_config <- createForestModelConfig(
    feature_types = feature_types,
    num_trees = num_trees,
    min_samples_leaf = 5,
    num_features = p,
    num_observations = n,
    variable_weights = variable_weights,
    leaf_dimension = leaf_dimension,
    leaf_model_type = outcome_model_type
)
global_model_config <- createGlobalModelConfig(
    global_error_variance = sigma2_init
)

# Forest model object
forest_model <- createForestModel(
    forest_dataset,
    forest_model_config,
    global_model_config
)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
forest_samples <- createForestSamples(num_trees, 1, T)
active_forest <- createForest(num_trees, 1, T)

# Initialize the leaves of each tree in the forest
leaf_init <- mean(y_standardized)
active_forest$prepare_for_sampler(
    forest_dataset,
    outcome,
    forest_model,
    outcome_model_type,
    leaf_init
)

# Prepare to run the sampler
num_burnin <- 3000
num_mcmc <- 1000
sigma2_samples <- rep(NA, num_mcmc)
a2_samples <- rep(NA, num_mcmc)
tau2_samples <- rep(NA, num_mcmc)
phi_i_samples <- matrix(NA, n, num_mcmc)
rmse_samples <- rep(0, num_mcmc)
fhat_samples <- matrix(0, n, num_mcmc)
current_sigma2 <- sigma2_init
current_a2 <- a2_init
current_tau2 <- tau2_init
current_phi_i <- phi_i_init

# Run the MCMC sampler
for (i in 1:(num_burnin + num_mcmc)) {
    keep_sample <- i > num_burnin

    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset,
        outcome,
        forest_samples,
        active_forest,
        rng,
        forest_model_config,
        global_model_config,
        keep_forest = keep_sample,
        gfr = F, 
        num_threads = 1
    )

    # Sample local variance parameters
    current_phi_i <- sample_phi_i(
        y_standardized,
        forest_dataset,
        active_forest,
        current_a2,
        current_tau2,
        nu
    )

    # Sample a2
    current_a2 <- sample_a2(
        y_standardized,
        forest_dataset,
        active_forest,
        current_phi_i
    )
    if (keep_sample) {
        a2_samples[i - num_burnin] <- current_a2 * y_std^2
    }

    # Sample tau2
    current_tau2 <- sample_tau2(current_phi_i, nu)
    if (keep_sample) {
        tau2_samples[i - num_burnin] <- current_tau2 * y_std^2
        sigma2_samples[i - num_burnin] <- current_tau2 * current_a2 * y_std^2
    }

    # Update observation-specific variance weights
    forest_dataset$update_variance_weights(current_phi_i * current_a2)

    # Compute in-sample RMSE and cache mean function samples
    if (keep_sample) {
        yhat_forest <- active_forest$predict(forest_dataset) * y_std + y_bar
        error <- (m_x - yhat_forest)
        rmse_samples[i - num_burnin] <- sqrt(mean(error * error))
        fhat_samples[, i - num_burnin] <- yhat_forest
    }
}

# Compute posterior mean of conditional expectations for the non-robust model
m_x_hat_posterior_mean <- rowMeans(fhat_samples)

# Now prepare to run the same sampler without robust errors for comparison
forest_dataset <- createForestDataset(X)
outcome <- createOutcome(y_standardized)

# Random number generator (std::mt19937)
rng <- createCppRNG(random_seed)

# Model configuration
outcome_model_type <- 0
leaf_dimension <- 1
num_trees <- 200
feature_types <- as.integer(rep(0, p)) # 0 = numeric
variable_weights <- rep(1 / p, p)
forest_model_config <- createForestModelConfig(
    feature_types = feature_types,
    num_trees = num_trees,
    num_features = p,
    min_samples_leaf = 5,
    num_observations = n,
    variable_weights = variable_weights,
    leaf_dimension = leaf_dimension,
    leaf_model_type = outcome_model_type
)
global_model_config <- createGlobalModelConfig(
    global_error_variance = sigma2_init
)

# Forest model object
forest_model <- createForestModel(
    forest_dataset,
    forest_model_config,
    global_model_config
)

# "Active forest" (which gets updated by the sample) and
# container of forest samples (which is written to when
# a sample is not discarded due to burn-in / thinning)
forest_samples <- createForestSamples(num_trees, 1, T)
active_forest <- createForest(num_trees, 1, T)

# Initialize the leaves of each tree in the forest
leaf_init <- mean(y_standardized)
active_forest$prepare_for_sampler(
    forest_dataset,
    outcome,
    forest_model,
    outcome_model_type,
    leaf_init
)
active_forest$adjust_residual(forest_dataset, outcome, forest_model, F, F)

# Prepare to run the sampler
global_var_samples <- rep(NA, num_mcmc)
rmse_samples_non_robust <- rep(0, num_mcmc)
fhat_samples_non_robust <- matrix(0, n, num_mcmc)
current_sigma2 <- sigma2_init

# Run the MCMC sampler
for (i in 1:(num_burnin + num_mcmc)) {
    keep_sample <- i > num_burnin

    # Sample forest
    forest_model$sample_one_iteration(
        forest_dataset,
        outcome,
        forest_samples,
        active_forest,
        rng,
        forest_model_config,
        global_model_config,
        keep_forest = keep_sample,
        gfr = F, 
        num_threads = 1
    )

    # Sample global error variance parameter
    current_sigma2 <- sampleGlobalErrorVarianceOneIteration(
        outcome,
        forest_dataset,
        rng,
        1,
        1
    )
    global_model_config$update_global_error_variance(current_sigma2)
    if (keep_sample) {
        global_var_samples[i - num_burnin] <- current_sigma2 * y_std^2
    }

    # Compute in-sample RMSE
    if (keep_sample) {
        yhat_forest <- active_forest$predict(forest_dataset) * y_std + y_bar
        error <- (m_x - yhat_forest)
        rmse_samples_non_robust[i - num_burnin] <- sqrt(mean(error * error))
        fhat_samples_non_robust[, i - num_burnin] <- yhat_forest
    }
}

# Plot RMSE samples side-by-side
pdf(
    "figures/R/custom-interface-bart-robust-rmse-comparison.pdf",
    width = 4,
    height = 3,
    pointsize = 10
)
y_bounds <- range(c(rmse_samples, rmse_samples_non_robust))
y_bounds[2] <- y_bounds[2] * 1.25
plot(
    rmse_samples,
    type = "l",
    col = "blue",
    ylim = y_bounds,
    ylab = "In-Sample RMSE",
    xlab = "Iteration"
)
lines(rmse_samples_non_robust, col = "red")
legend(
    "topleft",
    legend = c("Gaussian Errors", "t Errors"),
    col = c("red", "blue"),
    lty = 1
)
dev.off()

# Compute posterior mean of conditional expectations for the non-robust model
m_x_hat_posterior_mean_non_robust <- rowMeans(fhat_samples_non_robust)

# Plot predicted versus actual for both functions
pdf(
    "figures/R/custom-interface-bart-robust-pred-actual-comparison.pdf",
    width = 4,
    height = 3,
    pointsize = 10
)
y_bounds <- range(m_x)
y_bounds[2] <- y_bounds[2] * 1.1
plot(
    m_x_hat_posterior_mean_non_robust,
    m_x,
    pch = 20,
    col = 'lightgray',
    xlab = 'Predicted f(x)',
    ylab = 'True f(x)',
    ylim = y_bounds
)
abline(0, 1)
points(m_x_hat_posterior_mean, m_x, pch = 20, cex = 0.5)
legend(
    "topleft",
    legend = c('Gaussian errors', 't errors'),
    pch = c(20, 20),
    col = c('lightgray', 'black')
)
dev.off()

##################################################################
### Section 6: Simulated Causal Inference Demo
##################################################################

# Data generation parameters
n <- 1000
p <- 5

# Generate covariates
X <- matrix(runif(n * p), nrow = n, ncol = p)

# Generate model components that are functions of X
mu_x <- prog_fn(X)
pi_x <- propensity_fn(X)
tau_x <- cate_fn(X)

# Generate treatment
Z <- rbinom(n, 1, pi_x)

# Generate outcome
E_Y_ZX <- mu_x + tau_x * Z
y <- E_Y_ZX + rnorm(n, 0, 1)

# Fit a propensity model
general_params_propensity = list(
  probit_outcome_model = T,
  sample_sigma2_global = F,
  random_seed = random_seed, 
  num_threads = 1
)
propensity_model <- bart(
  X_train = X,
  y_train = Z,
  general_params = general_params_propensity
)
propensity <- predict(
  propensity_model,
  X = X,
  type = "mean",
  terms = "y_hat"
)

# Fit a causal model
bcf_model <- bcf(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = propensity,
  num_gfr = 10,
  num_burnin = 2000,
  num_mcmc = 1000,
  general_params = list(random_seed = random_seed, 
                        num_threads = 1)
)

# Compute the posterior of the treatment effect function
tau_hat_posterior <- predict(
  bcf_model,
  X = X,
  Z = Z,
  propensity = propensity,
  type = "posterior",
  terms = "cate"
)
ate_posterior <- colMeans(tau_hat_posterior)

# Plot the true CATE against the CATE posterior mean
pdf(
  "figures/R/simulated-cate-true-fitted.pdf",
  width = 4,
  height = 3,
  pointsize = 8
)
plot(
  rowMeans(tau_hat_posterior),
  tau_x,
  xlab = "Estimated CATE",
  ylab = "True CATE"
)
abline(0, 1, col = "blue", lty = 3, lwd = 5)
dev.off()

# ATE histogram
pdf(
  "figures/R/simulated-ate-posterior.pdf",
  width = 4,
  height = 3,
  pointsize = 8
)
hist(
  ate_posterior,
  xlab = "ATE",
  main = NULL,
  breaks = 30
)
abline(v = mean(tau_x), col = "blue", lty = 1, lwd = 4)
dev.off()

# Fit a surrogate regression tree to the posterior mean of the CATE samples
summary_df <- data.frame(tau_hat = rowMeans(tau_hat_posterior), X)
cate <- rpart(tau_hat ~ ., summary_df, control = rpart.control(cp = 0.001))
cate <- prune.rpart(cate, cp = 0.05)

# Define separate colors for left and rightmost nodes
plot.cart <- function(rpart.obj) {
  rpart.frame <- rpart.obj$frame
  left <- which.min(rpart.frame$yval)
  right <- which.max(rpart.frame$yval)
  nodes <- rep(NA, nrow(rpart.frame))
  for (i in 1:length(nodes)) {
    if (rpart.frame$yval[i] == rpart.frame$yval[right]) {
      nodes[i] <- "gold2"
    } else if (rpart.frame$yval[i] == rpart.frame$yval[left]) {
      nodes[i] <- "tomato3"
    } else {
      nodes[i] <- "lightblue3"
    }
  }
  return(nodes)
}

# Plot the surrogate regression tree
pdf(
  "figures/R/simulated-cate-tree.pdf",
  width = 3.5,
  height = 3,
  pointsize = 10
)
rpart.plot(cate, main = "", box.col = plot.cart(cate))
dev.off()

##################################################################
### Section 7: ACIC Data Causal Inference Demo
##################################################################

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
    random_seed = random_seed, 
    num_threads = 1
)
propensity_model <- bart(
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
bcf_model <- bcf(
    X_train = covariate_df,
    Z_train = Z,
    y_train = y,
    propensity_train = propensity,
    num_gfr = 10,
    num_burnin = 2000,
    num_mcmc = 1000,
    general_params = list(random_seed = random_seed, num_threads = 1),
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
bcf_model_rfx <- bcf(
    X_train = covariate_df,
    Z_train = Z,
    y_train = y,
    propensity_train = propensity,
    rfx_group_ids_train = group_ids,
    num_gfr = 10,
    num_burnin = 2000,
    num_mcmc = 1000,
    general_params = list(random_seed = random_seed, num_threads = 1),
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
