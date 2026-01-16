# Load libraries
library(stochtree)

# Seed for reproducibility
random_seed <- 1234
set.seed(random_seed)

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

### Helper functions for custom sampler

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
        gfr = F
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
