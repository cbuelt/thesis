library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))

# Simulate Brown-Resnick model
simulate_train <- function(params){
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
no_cores <- detectCores() - 2
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('simulate_train'))
clusterEvalQ(cl, library(SpatialExtremes))

train_data <- parApply(cl, train_params, MARGIN = 1, FUN = simulate_train)
stopCluster(cl)


#Save train_data
save(train_data, file = "../data/data_test/train_data.RData")



#Test set
simulate <- function(n, range, smoothness){
  x <- seq(1,20, length = 25)
  coord <- cbind(x,x)
  brown <- rmaxstab(n = n, coord = coord, cov.mod = "brown", range = range, smoothness = smoothness, grid = TRUE)
  return(brown)
}

#Generate parameters
n <- 50
range_seq <- c(0.5, 0.75, 1, 1.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- dim(comb)[1]
test_params <- cbind(rep(comb$Var1, each = 50), rep(comb$Var2, each = 50))
#Save parameters
save(test_params, file = "../data/data_test/test_params.RData")


result <- array(data = 0, dim = c((n_comb),25,25,n))

for (i in 1:n_comb){
  range <- comb[i,1]
  smooth <- comb[i,2]
  sim <- simulate(n, range = range, smoothness = smooth)
  result[i,,,] <- sim
}

save(result, file = "../data/data_test/test_data.RData")




#x <- seq(0,10, length = 100)
#coord <- cbind(x,x)
schlather <- rmaxstab(1, coord, "powexp", nugget = 0, range = 0.5, smooth = 1, grid = TRUE)
levelplot(t(schlather[c(nrow(schlather):1),]), col.regions= terrain.colors(100))



