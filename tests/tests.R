library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)



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


# Simulate parameters
smooth <- runif(n = 1000, min = 0, max = 2)
range <- rexp(n = 1000, rate = 0.25)
params <- cbind(range, smooth)
plot(hist(range, breaks = 30))


