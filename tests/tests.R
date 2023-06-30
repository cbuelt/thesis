library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)
library(dplyr)



current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 1

length <- 25
x <- seq(0,1, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
range <- 1.2
smooth <- 1.05
brown <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = range, smooth = smooth)
schlather <- rmaxstab(n = 1, coord = grid, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth)

max(schlather)
max(brown)

image(x, x, array(brown[1,], dim = c(length,length)), col = terrain.colors(64))
image(x, x, array(schlather[1,], dim = c(length,length)), col = terrain.colors(64))

dis <- distance(grid)
dis_unique <- unique(dis)
dis[dis == dis_unique[10]]

#madogram
mado <- fmadogram(brown, grid)

n.site <- 40
n.obs <- 3
coord <- matrix(runif(2 * n.site, 0, 10), ncol = 2)
data <- rmaxstab(n.obs, coord, "gauss", cov11 = 1, cov12 = 0, cov22 = 1)
par(mfrow=c(1,2))
mado <- madogram(data, coord)


#Test own implementation of empirical madogram
length <- 5
x <- seq(0,10, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
data <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = 1.2, smooth = 1.5)
dis <- distance(grid)

#Compute all upper triangular distances
res <- array(data = 0, dim = choose(length **2, 2))
cnt <- 0
for (i in 1:(length**2-1)){
  for (j in (i+1):length**2){
    res[cnt] <- abs(data[i]- data[j])
    cnt <- cnt+1
  }
}

df <- data.frame(cbind(res, dis))
plot(x = df$dis, y = df$res)
df$dis <- round(df$dis,6)
mado <- df %>% group_by(dis) %>% summarise(res = 0.5*mean(res))

plot(mado)

#Real madogram
data <- rmaxstab(n = 2, coord = grid, cov.mod = "brown", range = 1.2, smooth = 1.5)
#fit <- fitmaxstab(data, grid, cov.mod = "brown")
#fit$fitted.values
mado <- madogram(data, grid)




# Compare train and test parameters
load("../data/exp_3/data/brown_test_params.RData")
load("../data/exp_3/data/brown_train_params.RData")

plot(train_params, col = "blue")
points(test_params, col = "red", pch = 16)


