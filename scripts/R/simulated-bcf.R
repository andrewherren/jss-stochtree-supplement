# Load libraries
library(stochtree)
library(rpart)
library(rpart.plot)

# Seed for reproducibility
random_seed <- 12345
set.seed(random_seed)

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
  random_seed = random_seed
)
propensity_model <- stochtree::bart(
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
bcf_model <- stochtree::bcf(
  X_train = X,
  Z_train = Z,
  y_train = y,
  propensity_train = propensity,
  num_gfr = 10,
  num_burnin = 2000,
  num_mcmc = 1000,
  general_params = list(random_seed = random_seed)
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
