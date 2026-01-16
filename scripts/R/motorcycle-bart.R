# Load stochtree
library(stochtree)
library(MASS)

# Seed for reproducibility
random_seed = 12345
set.seed(random_seed)

# Load the motorcycle data
mcycle <- MASS::mcycle
n <- nrow(mcycle)

# Run homoskedastic stochtree
num_gfr <- 10
num_burnin <- 0
num_mcmc <- 100
num_samples <- num_gfr + num_burnin + num_mcmc
general_params <- list(sample_sigma2_global = T, random_seed = random_seed)
bart_model <- stochtree::bart(
    X_train = as.matrix(mcycle$times),
    y_train = mcycle$accel,
    num_gfr = num_gfr,
    num_burnin = num_burnin,
    num_mcmc = num_mcmc,
    general_params = general_params
)
sigma_samples <- sqrt(bart_model$sigma2_global_samples)

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
general_params <- list(sample_sigma2_global = F, random_seed = random_seed)
variance_forest_params <- list(
    num_trees = 20,
    alpha = 0.5,
    beta = 3.0,
    min_samples_leaf = 20
)
bart_model_het <- stochtree::bart(
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
