library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- 25#detectCores() - 1

simulate <- function(params){
  length <- 25
  x <- seq(0,length, length = length)
  grid <- expand.grid(x,x)
  grid <- array(unlist(grid), dim = c(length**2,2))
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  if (model == "brown"){
    data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = range, smooth = smooth)
  }else{
    data <- rmaxstab(1, coord = grid, cov.mod = model, nugget = 0, range = range, smooth = smooth)
  }
  return(data)
}

#
# Training/Validation data
#

# Set parameters
n <- 1000
exp <- "exp_4_1"

# Simulate parameters
smooth <- runif(n = n, min = 0, max = 1.5)
range <- runif(n = n, min = 0.5, max = 3)
train_params <- cbind(range, smooth)

#Generate 5 pairs of parameters for aggregating
n_pair <- 5
train_params <- cbind(rep(range, each = n_pair), rep(smooth, each = n_pair))
colnames(train_params) <- c("range", "smooth")

for (model in c("brown", "powexp")){
  #Save params
  save(train_params, file = paste0("../data/",exp,"/data/", model, "_train_params.RData"))
  #Create train set
  # Initiate cluster
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('simulate', 'model'))
  clusterEvalQ(cl, library(SpatialExtremes))
  
  train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
  stopCluster(cl)
  
  #Save train_data
  save(train_data, file = paste0("../data/",exp,"/data/", model, "_train_data.RData"))
}


# Create test dataset
# Set parameters
n_each <- 20 * n_pair
n <- 25
exp <- "exp_4_1"

# Simulate parameters
smooth <- rep(runif(n = n, min = 0, max = 1.5), each = n_each)
range <- rep(runif(n = n, min = 0.5, max = 3), each = n_each)

#range <- c(0.2748737, 0.4687288, 3.575283, 4.846655, 1.362708, 1.525501, 4.017220, 3.238309)
#smooth<- c(0.0845058, 1.069622, 0.6416916, 0.3100938, 1.899021, 1.984236, 1.613812, 1.793770)
#range <- rep(range, each = n_each)
#smooth <- rep(smooth, each = n_each)
test_params <- cbind(range, smooth)

for (model in c("brown", "powexp")){
  #Save params
  save(test_params, file = paste0("../data/",exp,"/data/", model, "_test_params.RData"))
  #Create train set
  # Initiate cluster
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('simulate', 'model'))
  clusterEvalQ(cl, library(SpatialExtremes))
  
  test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
  stopCluster(cl)
  
  #Save train_data
  save(test_data, file = paste0("../data/",exp,"/data/", model, "_test_data.RData"))
}
