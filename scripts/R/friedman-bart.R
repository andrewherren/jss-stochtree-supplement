# Load libraries
library(stochtree)

# Seed for reproducibility
random_seed <- 12345
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
xbart_model <- stochtree::bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 20,
    num_mcmc = 0,
    general_params = list(random_seed = random_seed)
)
xbart_json <- saveBARTModelToJsonString(xbart_model)
mean_forest_params = list(
    alpha = 0.25,
    beta = 2,
    min_samples_leaf = 10,
    max_depth = 8
)
bart_model <- stochtree::bart(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    num_gfr = 0,
    num_burnin = 0,
    num_mcmc = 10000,
    mean_forest_params = mean_forest_params,
    previous_model_json = xbart_json,
    previous_model_warmstart_sample_num = 20,
    general_params = list(random_seed = random_seed)
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
