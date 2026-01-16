# Load libraries
library(stochtree)

# Seed for reproducibility
random_seed <- 1025
set.seed(random_seed)

# Helper function to generate "Friedman 1" dataset
friedman_mean <- function(x) {
    10 *
        sin(pi * x[, 1] * x[, 2]) +
        20 * (x[, 3] - 0.5)^2 +
        10 * x[, 4] +
        5 * x[, 5]
}

# Functions to sample variance parameters
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

# Updated example with robust errors
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
        gfr = F
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
        gfr = F
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
