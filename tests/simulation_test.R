library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)

current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))

# Simulate Brown-Resnick model
simulate_train <- function(n, range_seq, smooth_seq){
  x <- seq(1,20, length = 25)
  coord <- cbind(x,x)
  range <- runif(1, min = range_seq[1], max = range_seq[2])
  smooth <- runif(1, min = smooth_seq[1], max = smooth_seq[2])
  brown <- rmaxstab(n = n, coord = coord, cov.mod = "brown", range = range, smoothness = smooth, grid = TRUE)
  return(list(data = brown, range = range, smooth = smooth))
}


#Create train set
no_cores <- detectCores() - 2
# Initiate cluster
cl <- makeCluster(no_cores)
clusterEvalQ(cl, library(SpatialExtremes))

n <- 2000
train_set <- parLapply(cl, rep(1,n), simulate_train, range_seq = c(0.1,3), smooth_seq = c(0.5,1.9))
stopCluster(cl)

# Create data outputs
train_var <- data.frame(matrix(data = 0, nrow = n, ncol = 2))
colnames(train_var) <- c("range", "smoothness")
train_data <- data.frame(matrix(data = 0, nrow = n, ncol = 625))
for (i in 1:n){
  train_var[i,] <- c(train_set[[i]]$range, train_set[[i]]$smooth)
  train_data[i,] <- array(train_set[[i]]$data)
}

write.csv(train_var, file = "sim_data/train_var.csv")
write.csv(train_data, file = "sim_data/train_data.csv")



x <- seq(1,20, length = 25)
grid <- expand.grid(x=x, y=x)

rand <- seq(1,4)
grid$"a" <- array(train_set[[rand[1]]]$data)
grid$"b" <- array(train_set[[rand[2]]]$data)
grid$"c" <- array(train_set[[rand[3]]]$data)
grid$"d" <- array(train_set[[rand[4]]]$data)


x1 <- levelplot(a~x*y, grid, col.regions= terrain.colors(100), main = paste0(round(train_var[1,1],4),"-", round(train_var[1,2],4)))
x2 <- levelplot(b~x*y, grid, col.regions= terrain.colors(100), main = paste0(round(train_var[2,1],4),"-", round(train_var[2,2],4)))
x3 <- levelplot(c~x*y, grid, col.regions= terrain.colors(100), main = paste0(round(train_var[3,1],4),"-", round(train_var[3,2],4)))
x4 <- levelplot(d~x*y, grid, col.regions= terrain.colors(100), main = paste0(round(train_var[4,1],4),"-", round(train_var[4,2],4)))

grid.arrange(x1, x2, x3, x4, ncol=2)



#Test set
simulate <- function(n, range, smoothness){
  x <- seq(1,20, length = 25)
  coord <- cbind(x,x)
  brown <- rmaxstab(n = n, coord = coord, cov.mod = "brown", range = range, smoothness = smoothness, grid = TRUE)
  return(brown)
}
range_seq <- c(0.5, 0.75, 1, 1.5)
smooth_seq <- c(0.8,1.05,1.3,1.55)
comb <- expand.grid(range_seq, smooth_seq)
n_comb <- length(range_seq) * length(smooth_seq)
n <- 50
results <- data.frame(matrix(data = 0, nrow = (n_comb * n), ncol = (2+25**2)))
for (i in 1:n_comb){
  range <- comb[i,1]
  smooth <- comb[i,2]
  sim <- simulate(n, range = range, smoothness = smooth)
  data <- as.matrix(data.frame(data = aperm(sim, c(3,1,2))))
  results[(1+50*(i-1)):(50*i),] <- as.data.frame(cbind(rep(range, n), rep(smooth, n), data))
}

write.csv(results, file = "sim_data/test_data.csv")

c <- 20
x <- seq(1,20, length = 25)
grid <- expand.grid(x=x, y=x)
res <- t(results[c,3:627])
grid$z <- res
smooth <- results[c,1]
range <- results[c,2]
levelplot(z~x*y, grid, col.regions= terrain.colors(100), main = paste0(round(smooth,4),"-", round(range,4)))





#x <- seq(0,10, length = 100)
#coord <- cbind(x,x)
schlather <- rmaxstab(1, coord, "powexp", nugget = 0, range = 0.5, smooth = 1, grid = TRUE)
levelplot(t(schlather[c(nrow(schlather):1),]), col.regions= terrain.colors(100))



