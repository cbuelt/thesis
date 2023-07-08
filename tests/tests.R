library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)
library(dplyr)



current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 2

length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
field <- rmaxstab(n = 50, coord = grid, cov.mod = "brown", range = 2, smooth = 1.6)

#Input triplet and maxstab data and get coefficient
pairwise_ext_coef <- function(i, j, data){
  n <- dim(data)[1]
  x_1 <-data[,i]
  x_2 <-data[,j]
  max <- apply(cbind(x_1,x_2), FUN = max, MARGIN = 1)
  return(n/sum(1/max))
}


#Generate vector with saved distances and extreal coefficient 
result <- array(data = NA, dim = c(choose(length**2, 2), 2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    h <- dist(rbind(grid[i,], grid[j,]))
    coeff <- pairwise_ext_coef(i,j,field)
    result[cnt,] <- c(h,coeff)
    cnt <- cnt +1
  }
}

agg <- aggregate(result[,2], list(round(result[,1],5)), FUN=mean)
names(agg) <- c("h", "Coeff")

#Plot extremal coefficient

brown_ext_coeff <- function(h, range, smooth){
  var <- (h/range)**smooth
  coeff <- 2*pnorm(sqrt(var)/2)
  return(coeff)
}


valid <- data.frame(cbind(agg[,1],brown_ext_coeff(agg[,1],2,1.6)))
names(valid) <- c("h", "Coeff")


plot(agg, ylim = c(0.5,2.5))
lines(valid, col = "red")

# Schlather Model
field <- rmaxstab(n = 5, coord = grid, cov.mod = "powexp", nugget = 0, range = 1.5, smooth = 0.8)
result <- array(data = NA, dim = c(choose(length**2, 2), 2))
cnt <- 1
for (i in 1:624){
  for (j in (i+1):625){
    h <- dist(rbind(grid[i,], grid[j,]))
    coeff <- pairwise_ext_coef(i,j,field)
    result[cnt,] <- c(h,coeff)
    cnt <- cnt +1
  }
}

agg <- aggregate(result[,2], list(round(result[,1],5)), FUN=mean)
names(agg) <- c("h", "Coeff")

#Plot extremal coefficient

schlather_ext_coeff <- function(h, range, smooth){
  corr <- exp(-(h/range)**smooth)
  coeff <- 1+ sqrt((1-corr)/2)
  return(coeff)
}


valid <- data.frame(cbind(agg[,1],schlather_ext_coeff(agg[,1],1.5,0.8)))
names(valid) <- c("h", "Coeff")


plot(agg, ylim = c(0.5,5))
lines(valid, col = "red")




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


