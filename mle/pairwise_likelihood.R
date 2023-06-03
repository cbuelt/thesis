library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 1

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
    cnt <- cnt +1
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
load(paste0("../data/exp_1/data/", model, "_test_data.RData"))
load(paste0("../data/exp_1/data/", model, "_test_params.RData"))
x <- seq(1,20, length = 25)
coord <- cbind(x,x)
n_test <- dim(test_params)[1]


run_start_values <- function(params, data, n_out){
  n <- dim(params)[1]
  mylist <- vector(mode = "list", length = n)
  names(mylist) <- seq(1,n)
  ll_list <- array(data = 0, dim = n)
 
  #Run optimization 
  for (i in 1:n){
    mylist[[i]] <- fitmaxstab(data = data, coord = coord, method = "L-BFGS-B", cov.mod = "brown",
                              start = list("range" = params[i,1], "smooth" = params[i,2]), 
                              weights = weights)
    ll_list[i] <- mylist[[i]]$opt.value
  }
  index <- sort(ll_list, decreasing = TRUE, index.return = TRUE)$ix[1:n_out]
  
  #Get new starting values
  new_params <- array(data = 0, dim = c(n_out,2))
  for (i in 1:n_out){
    new_params[i,] <- mylist[[i]]$fitted.values
  }
  return(new_params)
}


apply_mle <- function(i, data, params, n_sim = 20){
  test <- array(data = data[,i], dim = c(25,25))
  true <- params[i,]
  
  # Simulate similar parameters
  range_seq = c(0.9*true[1], 1.1*true[1])
  smooth_seq = c(0.9*true[2], 1.1*true[2])
  
  # Simulate uniform range
  range <- runif(n_sim, min = range_seq[1], max = range_seq[2])
  smooth <- runif(n_sim, min = smooth_seq[1], max = smooth_seq[2])
  params <- cbind(range,smooth)
  
  #Do prerun
  base_params <- run_start_values(params, test, 5)
  final_param <- run_start_values(base_params, test, 1)
  return(final_param)
}

#Create weights
weights <- array(data = NA, dim = 300)
cnt <- 1
for (i in 1:24){
  for (j in (i+1):25){
    if (abs(i-j) <=3){
      weights[cnt] <- 1
    }else{
      weights[cnt] <- 0
    }
    cnt <- cnt +1
  }
}

# Initiate cluster
cl <- makeCluster(no_cores)
clusterExport(cl,c('run_start_values', "coord", "weights"))
clusterEvalQ(cl, library(SpatialExtremes))

res <- parSapply(cl, seq(1, n_test), test_data, test_params, FUN = apply_mle)
stopCluster(cl)

# Convert to matrix and save
results <- t(res)
#Save
save(results, file = paste0("../data/exp_1/results/", model, "_mle2.RData"))




