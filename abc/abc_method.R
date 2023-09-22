current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Load functions
source("abc_functions.R")
#Get nodes
no_cores <-25#detectCores() - 1






#Steps
#Define grid
exp <- "exp_5"
model <- "powexp"
n <- 1
length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))


#Downsampling
downsample_size <- 5
x_small <- seq(0,length, length = downsample_size)
grid_small <- expand.grid(x_small,x_small)
grid <- array(unlist(grid_small), dim = c(downsample_size**2,2))

#Load data
load(paste0("../data/", exp,"/data/", model, "_test_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params.RData"))


#Measure time
start_time <- Sys.time()

# Calculate distance matrix
cluster_res <- get_clusters(grid = grid, n_stations = dim(grid)[1], method = "full")
print("Time for distance matrix:")
print(Sys.time()-start_time)

#Run simulations
index <- seq(1, 50, 1)
result <- sapply(X = index, FUN = abc_wrapper, simplify = "array", grid, cluster_res, model, 50000, n_cores = no_cores, interp = "interpolate", n_sim_each = 25,
                 q_filter = 0.01)
#Save results
save(result, file = paste0("../data/",exp,"/results/", model, "_abc_samples_interpolated_6.RData"))
print(dim(result))
print(Sys.time()-start_time)



