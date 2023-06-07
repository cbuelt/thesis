library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)



current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- 20 #detectCores() - 1

length <- 25
x <- seq(0,1, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
range <- 29#1.2
smooth <- 1.05
brown <- rmaxstab(n = 2, coord = grid, cov.mod = "brown", range = range, smooth = smooth)
schlather <- rmaxstab(n = 2, coord = grid, cov.mod = "powexp", nugget = 0, range = range, smooth = smooth)

max(schlather)
max(brown)

image(x, x, array(brown[1,], dim = c(length,length)), col = terrain.colors(64))
image(x, x, array(schlather[1,], dim = c(length,length)), col = terrain.colors(64))


# Simulate parameters
smooth <- runif(n = 1000, min = 0, max = 2)
range <- rexp(n = 1000, rate = 0.25)
params <- cbind(range, smooth)
plot(hist(range, breaks = 30))


