library(SpatialExtremes)
library(parallel)
library(gridExtra)
library(usedist)
library(combinat)


current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <-10#detectCores() - 1


n <- 1
length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
#Generate maxstab
field <- rmaxstab(n = n, coord = grid, cov.mod = "brown", range = 1, smoothness = 1)


# Add function
#d <- dim(grid)[1]
#Create (D 3) combinations
#triplets <- array(data = 0, dim = c(choose(d,3),3))
#Fill triplets
#index_comb <- combn(d,3)
#for (i in 1:choose(d,3)){
#  idx = index_comb[,i]
#  triplets[i,1] <- dist((grid[idx[1],]-grid[idx[2],]))
#  triplets[i,2] <- dist((grid[idx[1],]-grid[idx[3],]))
#  triplets[i,3] <- dist((grid[idx[2],]-grid[idx[3],]))
#}

#Approximate triplets with sampling
sample_dim <- 2000
triplets <- array(data = 0, dim = c(sample_dim, 3))
index_comb <- array(data = 0, dim = c(3,sample_dim))
for (i in 1:sample_dim){
  idx <- sample(x = 25, size = 3, replace = FALSE)
  index_comb[,i] <- idx
  triplets[i,1] <- dist((grid[idx[1],]-grid[idx[2],]))
  triplets[i,2] <- dist((grid[idx[1],]-grid[idx[3],]))
  triplets[i,3] <- dist((grid[idx[2],]-grid[idx[3],]))
}



distance_function <- function(a,b){
  return(sum(abs(a-b)))
}

triplet_distance <- function(v1,v2){
  a <- v1
  b <- permn(v2)
  dis <- sapply(b, FUN = distance_function, a)
  return(min(dis))
}

#Compute matrix
t <- dist_make(triplets, triplet_distance)
#Clustering
cluster <- hclust(t, method = "ward.D")
memb <- cutree(cluster, k = 100)

#Input triplet and maxstab data and get coefficient
triplet_ext_coef <- function(idx, data){
  n <- dim(data)[1]
  x_1 <-data[,idx[1]]
  x_2 <-data[,idx[2]]
  x_3 <-data[,idx[3]]
  max <- apply(cbind(x_1,x_2,x_3), FUN = max, MARGIN = 1)
  return(n/sum(1/max))
}

#Calculate ext coeff
ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, field)
#Aggregate per cluster
theta <- aggregate(ext_coeff, by = list(memb), FUN = mean)
plot(theta)


# Run actual sampling
get_abc_sample <- function(params, n, grid, memb, theta){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  sim_field <- rmaxstab(n = n, coord = grid, cov.mod = "brown", range = range, smoothness = smooth)
  #Calculate ext coeff
  ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, sim_field)
  #Aggregate per cluster
  theta_est <- aggregate(ext_coeff, by = list(memb), FUN = mean)
  #Calculate distance
  dist <- distance_function(theta,theta_est)
  return(dist)
  
}

n_sim <- 1000
n_sim_each <- 50
smooth <- runif(n = n_sim, min = 0, max = 2)
range <- runif(n = n_sim, min = 0, max = 10)
test_params <- cbind(range, smooth)

#Calculate parallel ABC
cl <- makeCluster(no_cores)
clusterExport(cl,c('get_abc_sample', 'triplet_ext_coef', 'index_comb', 'distance_function'))
clusterEvalQ(cl, library(SpatialExtremes))

t1 <- Sys.time()
dist_est <- parApply(cl, test_params, MARGIN = 1, FUN = get_abc_sample, n_sim_each, grid, memb, theta)
stopCluster(cl)
result <- cbind(test_params, dist_est)
print(Sys.time()-t1)

#Filter results
q <- quantile(result[,"dist_est"], 0.02)
result <- result[result[, "dist_est"] <= q, ]
result
colMeans(result)









## 2D index to 1D index
f <- function (i, j, dist_obj) {
  if (!inherits(dist_obj, "dist")) stop("please provide a 'dist' object")
  n <- attr(dist_obj, "Size")
  valid <- (i >= 1) & (j >= 1) & (i > j) & (i <= n) & (j <= n)
  k <- (2 * n - j) * (j - 1) / 2 + (i - j)
  k[!valid] <- NA_real_
  k
}

## 1D index to 2D index
finv <- function (k, dist_obj) {
  if (!inherits(dist_obj, "dist")) stop("please provide a 'dist' object")
  n <- attr(dist_obj, "Size")
  valid <- (k >= 1) & (k <= n * (n - 1) / 2)
  k_valid <- k[valid]
  j <- rep.int(NA_real_, length(k))
  j[valid] <- floor(((2 * n + 1) - sqrt((2 * n - 1) ^ 2 - 8 * (k_valid - 1))) / 2)
  i <- j + k - (2 * n - j) * (j - 1) / 2
  cbind(i, j)
}



