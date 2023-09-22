# Set path
current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
# Source functions
source("pl_functions.R")
#Get nodes
no_cores <- 25#detectCores() - 1


#
## Run single fit with choosing best likelihood
#
model <- "brown"
exp <- "final"
load(paste0("../data/", exp, "/data/", model, "_test_data.RData"))
load(paste0("../data/", exp, "/data/", model, "_test_params.RData"))

# Define grid used in data
length <- 30
x <- seq(0, 10, length = length)
grid <- expand.grid(x, x)
grid <- array(unlist(grid), dim = c(length ** 2, 2))
n_param <- dim(test_params)[1]

# Get weights
weights <- get_weights(grid, length)

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl, c('run_start_values', "model", "length"))
clusterEvalQ(cl, library(SpatialExtremes))

# Run estimations
res <-
  parSapply(cl, seq(1, n_param), test_data, test_params, grid, weights, FUN = apply_mle)
stopCluster(cl)
results <- t(res)
save(results,
     file = paste0(
       "../data/",
       exp,
       "/results/",
       model,
       "_single_image_fit_opt.RData"
     ))


# For fitting multiple images
#un <- unique(test_params, dim = 1)
#range <- rep(un[,1], each = 20)
#smooth <- rep(un[,2], each = 20)
#test_params <- cbind(range, smooth)

