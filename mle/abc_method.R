library(SpatialExtremes)
library(parallel)
library(gridExtra)
library(usedist)
library(combinat)


current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <-25#detectCores() - 1


#Functions
distance_function <- function(a,b){
  return(sum(abs(a-b)))
}

triplet_distance <- function(v1,v2){
  a <- v1
  b <- permn(v2)
  dis <- sapply(b, FUN = distance_function, a)
  return(min(dis))
}

#Input triplet and maxstab data and get coefficient
triplet_ext_coef <- function(idx, data){
  n <- dim(data)[1]
  x_1 <-data[,idx[1]]
  x_2 <-data[,idx[2]]
  x_3 <-data[,idx[3]]
  max <- apply(cbind(x_1,x_2,x_3), FUN = max, MARGIN = 1)
  return(n/sum(1/max))
}

# Run actual sampling
get_abc_sample <- function(params, n, grid, model, memb, index_comb, theta){
  range <- params[["range"]]
  smooth <- params[["smooth"]]
  if (model=="brown"){
    sim_field <- rmaxstab(n = n, coord = grid, cov.mod = model, range = range, smooth = smooth)
  }else{
    sim_field <- rmaxstab(n = n, coord = grid, cov.mod = model, nugget = 0,  range = range, smooth = smooth)
  }
  #Calculate ext coeff
  ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, sim_field)
  #Aggregate per cluster
  theta_est <- aggregate(ext_coeff, by = list(memb), FUN = mean)
  #Calculate distance
  dist <- distance_function(theta,theta_est)
  return(dist)
}

# Get data based on index
get_data <- function(i, type = "full"){
  data <- test_data[,i]
  if(type != "full"){
    #If interpolation
    break
  }else{
    result <- array(data, dim = c(1, length(data)))
    return(result)
  }
}

get_clusters <- function(grid, n_stations, n_cluster = 100, method = "full", approx_dim = NULL){
  # Calculate distance matrix for grid
  if(method == "full"){
    triplets <- array(data = 0, dim = c(choose(n_stations,3),3))
    #Fill triplets
    index_comb <- combn(n_stations,3)
    for (i in 1:choose(n_stations,3)){
      idx = index_comb[,i]
      triplets[i,1] <- dist((grid[idx[1],]-grid[idx[2],]))
      triplets[i,2] <- dist((grid[idx[1],]-grid[idx[3],]))
      triplets[i,3] <- dist((grid[idx[2],]-grid[idx[3],]))
    }
  }else{
    #Approximate triplets with sampling
    triplets <- array(data = 0, dim = c(approx_dim, 3))
    index_comb <- array(data = 0, dim = c(3,approx_dim))
    for (i in 1:approx_dim){
      idx <- sample(x = n_stations, size = 3, replace = FALSE)
      index_comb[,i] <- idx
      triplets[i,1] <- dist(rbind(grid[idx[1],],grid[idx[2],]))
      triplets[i,2] <- dist(rbind(grid[idx[1],],grid[idx[3],]))
      triplets[i,3] <- dist(rbind(grid[idx[2],],grid[idx[3],]))
    }
  }
  
  #Compute distance matrix
  dist_matrix <- dist_make(triplets, triplet_distance)
  #Clustering
  cluster <- hclust(dist_matrix, method = "ward.D")
  members <- cutree(cluster, k = n_cluster)
  return(list("members" = members, "index" = index_comb))
}


run_abc_sampling <- function(data, grid, cluster_res, model, n_sim,
                             n_cores = 20, n_sim_each = 50, q_filter = 0.02){
  memb <- cluster_res$members
  index_comb <- cluster_res$index
  
  #Calculate extremal coefficient
  ext_coeff <- apply(index_comb, MARGIN = 2, FUN = triplet_ext_coef, data)
  #Aggregate per cluster
  theta_true <- aggregate(ext_coeff, by = list(memb), FUN = mean)
  
  #Generate sampling parameters
  smooth <- runif(n = n_sim, min = 0, max = 2)
  range <- runif(n = n_sim, min = 0, max = 10)
  test_params <- cbind(range, smooth)
  
  #
  #Calculate parallel ABC
  #
  cl <- makeCluster(no_cores)
  clusterExport(cl,c('get_abc_sample', 'triplet_ext_coef', 'distance_function'))
  clusterEvalQ(cl, library(SpatialExtremes))
  
  dist_est <- parApply(cl, test_params, MARGIN = 1, FUN = get_abc_sample,
                       n_sim_each, grid, model, memb, index_comb, theta_true)
  stopCluster(cl)
  result <- cbind(test_params, dist_est)
  
  #Filter results
  q <- quantile(result[,"dist_est"], q_filter)
  result <- result[result[, "dist_est"] <= q,]
  return(result)
}

abc_wrapper <- function(index, grid, cluster_res, model, n_sim,
                        n_cores = 20, n_sim_each = 50, q_filter = 0.02){
  # Measure time
  t1 <- Sys.time()
  data <- get_data(index)
  result <- run_abc_sampling(data = data, grid = grid, cluster_res = cluster_res, model = model,
                             n_sim = n_sim, n_cores = n_cores, n_sim_each = n_sim_each, q_filter = q_filter)
  print(paste0("Total time: ", Sys.time()-t1))
  return(result)
}



#Steps
#Define grid
exp <- "exp_4"
model <- "brown"
n <- 1
length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
# Calculate distance matrix
cluster_res <- get_clusters(grid = grid, n_stations = length**2, method = "Sampling", approx_dim = 5000)

#Load data
load(paste0("../data/", exp,"/data/", model, "_test_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params.RData"))

#Run simulations
index <- seq(1, 750, 30)
result <- sapply(X = index, FUN = abc_wrapper, simplify = "array", grid, cluster_res, model, 10000)
#Save results
save(result, file = paste0("../data/",exp,"/results/", model, "_abc_samples.RData"))
result



