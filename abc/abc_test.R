current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Load functions
source("abc_functions.R")
#Get nodes
no_cores <-40#detectCores() - 1




#Steps
#Define grid
exp <- "final"
model <- "brown"
n <- 1
length <- 30
x <- seq(0,10, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))


#Downsampling
downsample_size <- 5
x_small <- seq(0,10, length = downsample_size)
grid_small <- expand.grid(x_small,x_small)
grid_small <- array(unlist(grid_small), dim = c(downsample_size**2,2))


# Generate 100 samples of same parameters
data <- array(0, dim = c(100,downsample_size**2))
for (i in 1:100){
  field <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = 0.5, smooth = 1)
  data_transformed <- array(field, dim = c(length, length))
  field_small <- bilinear.grid(x = x, y = x, z = data_transformed, nx = downsample_size, ny = downsample_size)
  data_small <- array(field_small$z, dim = c(1, downsample_size**2))
  data[i,] <- data_small[1,]
}

# Generate 1 sample of parameters
data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = 0.5, smooth = 1)

# Test grid
length <- 5
x <- seq(0,10, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
#data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = 0.5, smooth = 1)

#Measure time
start_time <- Sys.time()

# Calculate distance matrix
cluster_res <- get_clusters(grid = grid, n_stations = dim(grid)[1], method = "full")
print("Time for distance matrix:")
print(Sys.time()-start_time)


n_sim <- 25000
start_time <- Sys.time()
result <- run_abc_sampling(data = data, grid = grid, cluster_res = cluster_res, model = model,
                           n_sim = n_sim, n_cores = no_cores, n_sim_each = 100)
print("Time for simulation:")
print(Sys.time()-start_time)
# Save result example
save(result, file = "abc_example_several.RData")
result


plot(result[,1:2], pch = 16, col = "darkgreen", cex = 2)
points(test_params, col = "red", pch = 19, cex = 4)


#Save results
print(dim(result))
print(Sys.time()-start_time)
