current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Load functions
source("abc_functions.R")
#Get nodes
no_cores <-detectCores() - 6



#Steps
#Define grid
exp <- "outside_parameters"
model <- "brown"
# Define grid
length <- 30
x <- seq(0, length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))

# Define smaller grid for downsampling
downsample_size <- 5
x_small <- seq(0, 30, length = downsample_size)
grid_small <- expand.grid(x_small,x_small)
grid <- array(unlist(grid_small), dim = c(downsample_size**2,2))

#Load data
load(paste0("../data/", exp,"/data/", model, "_test_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_test_params.RData"))



#Measure time
start_time <- Sys.time()

#Run simulations
index <- seq(1, 250, 1)
result <- sapply(X = index, FUN = abc_wrapper, simplify = "array", grid, model, 50000, n_cores = no_cores, interp = "interpolate", n_sim_each = 25)
#Save results
save(result, file = paste0("../data/",exp,"/results/", model, "_abc_results.RData"))
print(dim(result))
print(Sys.time()-start_time)



