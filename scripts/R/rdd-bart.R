# Load libraries
library(stochtree)
library(rpart)
library(rpart.plot)

# Seed for reproducibility
random_seed <- 12345
set.seed(random_seed)

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
  random_seed = random_seed
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
bart_model <- stochtree::bart(
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
