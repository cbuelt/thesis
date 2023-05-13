library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 2

# Simulate Brown-Resnick model
simulate <- function(params){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  x <- seq(1,20, length = 25)
  coord <- cbind(x,x)
  brown <- rmaxstab(n = 1, coord = coord, cov.mod = "brown", range = range, smoothness = smooth, grid = TRUE)
  return(brown)
}
# Simulate training data
n <- 2000
range_seq = c(0.1,3)
smooth_seq = c(0.5,1.9)

# Simulate uniform range
range <- runif(n, min = range_seq[1], max = range_seq[2])
smooth <- runif(n, min = smooth_seq[1], max = smooth_seq[2])
train_params <- cbind(range,smooth)
#Save params
save(train_params, file = "../data/data_test/train_params.RData")
#Create train set
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate_train'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)


#Save train_data
save(train_data, file = "../data/data_test/train_data.RData")



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
save(test_params, file = "../data/data_test/test_params.RData")

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate_train'))
clusterEvalQ(cl, library(SpatialExtremes))

test_data <- parApply(cl, test_params, MARGIN = 1, FUN = simulate)
stopCluster(cl)

save(test_data, file = "../data/data_test/test_data.RData")





#x <- seq(0,10, length = 100)
#coord <- cbind(x,x)
schlather <- rmaxstab(1, coord, "powexp", nugget = 0, range = 0.5, smooth = 1, grid = TRUE)
levelplot(t(schlather[c(nrow(schlather):1),]), col.regions= terrain.colors(100))



