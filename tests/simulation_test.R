library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 1

# Simulate Brown-Resnick model
simulate <- function(params){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  x <- seq(1,20, length = 25)
  coord <- cbind(x,x)
  if (model == "brown"){
    data <- rmaxstab(n = 1, coord = coord, cov.mod = "brown", range = range, smooth = smooth, grid = TRUE)
  }else{
    data <- rmaxstab(1, coord = coord, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth, grid = TRUE)
  }
  
  return(data)
}




#
####### Brown-Resnick Model
#
model = "brown"

# Simulate training data
n <- 2000
range_seq = c(0.1,3)
smooth_seq = c(0.5,1.9)

# Simulate uniform range
range <- runif(n, min = range_seq[1], max = range_seq[2])
smooth <- runif(n, min = smooth_seq[1], max = smooth_seq[2])
train_params <- cbind(range,smooth)
#Save params
save(train_params, file = paste0("../data/exp_1/", model, "_train_params.RData"))
#Create train set
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', 'model'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

#Save train_data
save(train_data, file = paste0("../data/exp_1/", model, "_train_data.RData"))


#Test set
#Generate parameters
n <- 50
range_seq <- c(0.5, 0.75, 1, 1.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- dim(comb)[1]
test_params <- cbind(rep(comb$Var1, each = 50), rep(comb$Var2, each = 50))
colnames(test_params) <- c("range", "smooth")
#Save parameters
save(test_params, file = paste0("../data/exp_1/", model, "_test_params.RData"))

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', "model"))
clusterEvalQ(cl, library(SpatialExtremes))

test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

save(test_data, file = paste0("../data/exp_1/", model, "_test_data.RData"))







#
####### Schlather model
#
model = "schlather"

# Simulate training data
n <- 2000
range_seq = c(0.1,3)
smooth_seq = c(0.5,1.8)

# Simulate uniform range
range <- runif(n, min = range_seq[1], max = range_seq[2])
smooth <- runif(n, min = smooth_seq[1], max = smooth_seq[2])
train_params <- cbind(range,smooth)
#Save params
save(train_params, file = paste0("../data/exp_1/", model, "_train_params.RData"))
#Create train set
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', 'model'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

#Save train_data
save(train_data, file = paste0("../data/exp_1/", model, "_train_data.RData"))


#Test set
#Generate parameters
n <- 50
range_seq <- c(0.5, 1.5, 2, 2.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- dim(comb)[1]
test_params <- cbind(rep(comb$Var1, each = 50), rep(comb$Var2, each = 50))
colnames(test_params) <- c("range", "smooth")
#Save parameters
save(test_params, file = paste0("../data/exp_1/", model, "_test_params.RData"))

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate', "model"))
clusterEvalQ(cl, library(SpatialExtremes))

test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

save(test_data, file = paste0("../data/exp_1/", model, "_test_data.RData"))

