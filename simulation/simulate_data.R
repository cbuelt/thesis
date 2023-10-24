library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 4

simulate <- function(params, grid){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  if (model == "brown"){
    data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = range, smooth = smooth)
  }else if (model == "gauss"){
    data <- rmaxstab(n = 1, coord = grid, cov.mod = "gauss", cov11 = range, cov12 = 0, cov22 = range)
  }else{
    data <- rmaxstab(1, coord = grid, cov.mod = model, nugget = 0, range = range, smooth = smooth)
  }
  return(data)
}

#Set parameters
exp <- "application"
# Define grid
length <- 30
x <- seq(0, length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))

#
# Training/Validation data
#

# Set parameters
n <- 5000

# Simulate parameters
smooth <- runif(n = n, min = 0, max = 2)
range <- runif(n = n, min = 0, max = 5)
train_params <- cbind(range, smooth)


for (model in c("powexp")){
  #Save params
  save(train_params, file = paste0("../data/",exp,"/data/", model, "_train_params.RData"))
  #Create train set
  # Initiate cluster
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('simulate', 'model'))
  clusterEvalQ(cl, library(SpatialExtremes))
  
  train_data <- parApply(cl, train_params, grid, MARGIN = 1, FUN = simulate)
  stopCluster(cl)
  
  #Save train_data
  save(train_data, file = paste0("../data/",exp,"/data/", model, "_train_data.RData"))
}


# Create test dataset
# Set parameters
n_each <- 1
n <- 250

# Simulate parameters
smooth <- rep(runif(n = n, min = 0.3, max = 1.8), each = n_each)
range <- rep(runif(n = n, min = 0.5, max = 5), each = n_each)

#Outside parameters
smooth <- runif(n = 2000, min = 0, max = 2)
range <- runif(n = 2000, min = 0, max = 10)
test_params <- cbind(range, smooth)

test_params <- test_params[(test_params[,2] < 0.3) | (test_params[,2] > 1.8) | (test_params[,1] < 0.5) | (test_params[,1] > 5), ]
test_params <- test_params[1:n,]


for (model in c("brown")){
  #Save params
  save(test_params, file = paste0("../data/",exp,"/data/", model, "_test_params.RData"))
  #Create train set
  # Initiate cluster
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('simulate', 'model'))
  clusterEvalQ(cl, library(SpatialExtremes))
  
  test_data <- parApply(cl, test_params, grid, MARGIN = 1, FUN = simulate)
  stopCluster(cl)
  
  #Save train_data
  save(test_data, file = paste0("../data/",exp,"/data/", model, "_test_data.RData"))
}

