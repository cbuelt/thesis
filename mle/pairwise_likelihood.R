library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <-25#detectCores() - 1

#Test
#Create weights
weights <- array(data = NA, dim = choose(625,2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    if (abs(i-j) <=3){
      weights[cnt] <- 1
    }else{
      weights[cnt] <- 0
    }
    cnt <- cnt + 1
  }
}

range <- 2.5
smooth <- 0.8
start_range <- runif(1, 1.5,3)
start_smooth <- runif(1,0.5,1.7)
x <- seq(1,20, length = 25)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(625,2))
data <- rmaxstab(n = 3, coord = grid, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth)
fit <- fitmaxstab(data, grid, cov.mod = "powexp", method = "L-BFGS-B", weights = weights,
                  start = list("nugget" = 0, "range" = start_range, "smooth" = start_smooth))
print(fit$fitted.values)

#Fit with append
test <- array(rep(data[1,], each = 3), dim = c(3, 625))
fit_2 <- fitmaxstab(test, grid, cov.mod = "powexp", method = "L-BFGS-B", weights = weights,
                    start = list("nugget" = 0, "range" = start_range, "smooth" = start_smooth))
print(fit_2$fitted.values)






model <- "brown"
exp <- "exp_2"
load(paste0("../data/", exp,"/data/", model, "_test_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params.RData"))
x <- seq(1,20, length = 25)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(625,2))

# Estimate parameters from 50 simulations
n_param <- 16
n_test <- dim(test_params)[1] / n_param
true_params <- unique(test_params)
n_test = 50

#Weights
weights <- array(data = NA, dim = choose(625,2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    if (abs(i-j) <=3){
      weights[cnt] <- 1
    }else{
      weights[cnt] <- 0
    }
    cnt <- cnt +1
  }
}

param_range <- seq(n_param)

run_agg_fit <- function(i, data, grid, weights, n_test){
  data_subset <- t(data[,((i-1)*n_test+1):(i*n_test)])
  if (model == "brown"){
    fit <- fitmaxstab(data = data_subset, coord = grid, cov.mod = "brown", method = "L-BFGS-B", weights = weights)
  }else{
    fit <- fitmaxstab(data = data_subset, coord = grid, cov.mod = "powexp", method = "L-BFGS-B", weights = weights)
  }
  return(fit$fitted.values)
}

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('run_agg_fit', 'model'))
clusterEvalQ(cl, library(SpatialExtremes))

start.time <- Sys.time()
res <- parSapply(cl, param_range, test_data, grid, weights, n_test, FUN = run_agg_fit)
stopCluster(cl)
res <- t(res)
end.time <- Sys.time()
print(end.time - start.time)

save(res, file = paste0("../data/", exp,"/results/", model, "_fit_all_images.RData"))
res







#
######## Run one-image fit
#
model <- "powexp"
exp <- "exp_2"
load(paste0("../data/", exp,"/data/", model, "_test_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params.RData"))
x <- seq(1,20, length = 25)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(625,2))
n_param <- dim(test_params)[1]
#Weights
weights <- array(data = NA, dim = choose(625,2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    if (abs(i-j) <=3){
      weights[cnt] <- 1
    }else{
      weights[cnt] <- 0
    }
    cnt <- cnt +1
  }
}

single_image_fit <- function(i, params, data, grid, weights){
  data_subset <- array(rep(data[,i], each = 3), dim = c(3, 625))
  range <- params[i,1]
  smooth <- params[i,2]
  #Simulate starting values
  range_start <- runif(1, 0, 5)
  smooth_start <- runif(1, 0, 2)
  
  if (model == "brown"){
    fit <- fitmaxstab(data = data_subset, coord = grid, cov.mod = "brown", method = "L-BFGS-B", weights = weights,
                      start = list("range" = range_start, "smooth" = smooth_start))
  }else{
    fit <- fitmaxstab(data = data_subset, coord = grid, cov.mod = "powexp", method = "L-BFGS-B", weights = weights,
                      start = list("nugget" = 0, "range" = range_start, "smooth" = smooth_start))
  }
  return(fit$fitted.values)
}


range <- seq(n_param)
# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('single_image_fit', "model"))
clusterEvalQ(cl, library(SpatialExtremes))
res <- parSapply(cl, range, test_params, test_data, grid, weights, FUN = single_image_fit)
stopCluster(cl)
# Convert to matrix and save
results <- t(res)
#Save
save(results, file = paste0("../data/exp_2/results/", model, "_single_image_fit_wide.RData"))





#
## Run single fit with choosing best likelihood
#
model <- "powexp"
exp <- "exp_4_1"
load(paste0("../data/", exp,"/data/", model, "_test_data_outside.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params_outside.RData"))

# For fitting multiple images
un <- unique(test_params, dim = 1)
range <- rep(un[,1], each = 20)
smooth <- rep(un[,2], each = 20)
test_params <- cbind(range, smooth)


length <- 25
x <- seq(0,length, length = length**2)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(625,2))
n_param <- dim(test_params)[1]
#Weights
weights <- array(data = NA, dim = choose(625,2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    if (dist(grid[i,]-grid[j,]) <=5){
      weights[cnt] <- 1
    }else{
      weights[cnt] <- 0
    }
    cnt <- cnt +1
  }
}


run_start_values <- function(params, data, grid, weights, n_out){
  n <- dim(params)[1]
  mylist <- vector(mode = "list", length = n)
  names(mylist) <- seq(1,n)
  ll_list <- array(data = 0, dim = n)
  
  #Run optimization 
  for (i in 1:n){
    if (model == "brown"){
      mylist[[i]] <- fitmaxstab(data = data, coord = grid, method = "L-BFGS-B", cov.mod = "brown",
                                start = list("range" = params[i,1], "smooth" = params[i,2]), 
                                weights = weights)
    }else{
      mylist[[i]] <- fitmaxstab(data = data, coord = grid, method = "L-BFGS-B", cov.mod = model,
                                start = list("nugget" = 0, "range" = params[i,1], "smooth" = params[i,2]), 
                                weights = weights)
    }
    ll_list[i] <- mylist[[i]]$logLik
  }
  index <- sort(ll_list, decreasing = TRUE, index.return = TRUE)$ix[1:n_out]
  
  #Get new starting values
  param_dim <- length(mylist[[i]]$param)
  new_params <- array(data = 0, dim = c(n_out,param_dim))
  for (i in 1:n_out){
    new_params[i,] <- mylist[[i]]$fitted.values
  }
  return(new_params)
}


apply_mle <- function(i, data, params, grid, weights, n_sim = 10){
  #data_subset <- array(rep(data[,i], each = 3), dim = c(3, 625))
  #For multiple images
  data_subset <- t(data[,(i*5-4):(i*5)])
  
  true <- params[i,]
  
  # Simulate similar parameters
  range_seq = c(0.5,3)
  smooth_seq = c(0,1.5)
  
  # Simulate uniform range
  range <- runif(n_sim, min = range_seq[1], max = range_seq[2])
  smooth <- runif(n_sim, min = smooth_seq[1], max = smooth_seq[2])
  params <- cbind(range,smooth)
  
  #Do prerun
  base_params <- run_start_values(params, data_subset, grid, weights, 3)
  final_param <- run_start_values(base_params, data_subset, grid, weights, 1)
  return(final_param)
}


# Initiate cluster
start.time <- Sys.time()
cl <- makeCluster(no_cores)
clusterExport(cl,c('run_start_values', "model"))
clusterEvalQ(cl, library(SpatialExtremes))
res <- parSapply(cl, seq(1, n_param), test_data, test_params, grid, weights, FUN = apply_mle)
stopCluster(cl)
results <- t(res)
save(results, file = paste0("../data/",exp,"/results/", model, "_single_image_fit_opt.RData"))

