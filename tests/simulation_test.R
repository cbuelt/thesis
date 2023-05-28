library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- 20 #detectCores() - 1

simulate <- function(params){
  x <- seq(1,20, length = 25)
  grid <- expand.grid(x,x)
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  if (model == "brown"){
    data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = range, smooth = smooth)
  }else{
    grid <- array(unlist(grid), dim = c(625,2))
    data <- rmaxstab(1, coord = grid, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth)
  }
  return(data)
}

generate_params <- function(n, range_seq, smooth_seq){
  range <- runif(n, min = range_seq[1], max = range_seq[2])
  smooth <- runif(n, min = smooth_seq[1], max = smooth_seq[2])
  train_params <- cbind(range,smooth)
  return(train_params)
}

#Set general parameters
n <- 400
exp <- "exp_2"


#
####### Brown-Resnick Model
#
model = "brown"
type = "val"

# Simulate training data
range_seq = c(0.1,3)
smooth_seq = c(0.5,1.9)
train_params <- generate_params(n, range_seq, smooth_seq)


#Save params
save(train_params, file = paste0("../data/",exp,"/data/", model, "_", type, "_params.RData"))
#Create train set
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', 'model'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

#Save train_data
save(train_data, file = paste0("../data/",exp,"/data/", model, "_", type, "_data.RData"))


#Test set
#Generate parameters
n_test <- 50
range_seq <- c(0.5, 0.75, 1, 1.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- dim(comb)[1]
test_params <- cbind(rep(comb$Var1, each = n_test), rep(comb$Var2, each = n_test))
colnames(test_params) <- c("range", "smooth")
#Save parameters
save(test_params, file = paste0("../data/",exp,"/data/", model, "_test_params.RData"))

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', "model"))
clusterEvalQ(cl, library(SpatialExtremes))

test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

save(test_data, file = paste0("../data/",exp,"/data/", model, "_test_data.RData"))







#
####### Schlather model
#
model = "schlather"

# Simulate training data
range_seq = c(0.1,3)
smooth_seq = c(0.5,1.8)

# Simulate uniform range
range <- runif(n, min = range_seq[1], max = range_seq[2])
smooth <- runif(n, min = smooth_seq[1], max = smooth_seq[2])
train_params <- cbind(range,smooth)
#Save params
save(train_params, file = paste0("../data/",exp,"/data/", model, "_", type, "_params.RData"))
#Create train set
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', 'model'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

#Save train_data
save(train_data, file = paste0("../data/",exp,"/data/", model, "_", type, "_data.RData"))


#Test set
#Generate parameters
n_test <- 50
range_seq <- c(0.5, 1.5, 2, 2.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- dim(comb)[1]
test_params <- cbind(rep(comb$Var1, each = n_test), rep(comb$Var2, each = n_test))
colnames(test_params) <- c("range", "smooth")
#Save parameters
save(test_params, file = paste0("../data/",exp,"/data/", model, "_test_params.RData"))

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', "model"))
clusterEvalQ(cl, library(SpatialExtremes))

test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

save(test_data, file = paste0("../data/",exp,"/data/", model, "_test_data.RData"))



