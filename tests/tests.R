library(SpatialExtremes)
library(graphics)
library(lattice)
library(parallel)
library(gridExtra)
library(dplyr)
library(interp)
library(easyNCDF)



current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))
#Get nodes
no_cores <- detectCores() - 2


load("../data/outside_model/results/smith_abc_results.RData")
#names(dim(result)) <- c("abc_samples", "variable", "test_sample")
ArrayToNc(list(results = result), "../data/outside_model/results/smith_abc_results.nc")


#Load data
exp <- "exp_5"
model <- "powexp"
type <- "test"
load(paste0("../data/", exp,"/data/", model, "_", type, "_data.RData"))
load(paste0("../data/", exp,"/data/", model, "_", type, "_params.RData"))

length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))

#Downsampling
downsample_size <- 10
x_small <- seq(0,length, length = downsample_size)
grid_small <- expand.grid(x_small,x_small)
grid <- array(unlist(grid_small), dim = c(downsample_size**2,2))


# Get data based on index
get_data <- function(i, type = "full"){
  data <- test_data[,i]
  if(type != "full"){
    #If interpolation
    data_transformed <- array(data, dim = c(length, length))
    field_small <- bilinear.grid(x = x, y = x, z = data_transformed, nx = downsample_size, ny = downsample_size)
    data_small <- array(field_small$z, dim = c(1, downsample_size**2))
    return(data_small)
  }else{
    result <- array(data, dim = c(1, length(data)))
    return(result)
  }
}

n_params <- 500

filled.contour(x, x, array(test_data[,i], dim = c(length, length)), color.palette = terrain.colors, nlevels = 30)
test_data <- sapply(seq(1,n_params), get_data, type = "interp")

i <- 30


filled.contour(x_small, x_small, array(test_data[,i], dim = c(downsample_size, downsample_size)), color.palette = terrain.colors, nlevel = 30)

#Save
exp <- "exp_6"
save(test_data, file = paste0("../data/", exp,"/data/", model, "_test_data.RData"))



length_sample <- 6
field_transformed <- array(field, dim = c(length, length))

field_small <- bilinear.grid(x=x, y = x, z = field_transformed, nx = length_sample, ny = length_sample)

filled.contour(x, x, field_transformed, color.palette = terrain.colors, nlevels = 30)

filled.contour(field_small, color.palette = terrain.colors, nlevel = 30)

# Plot original 6x6 grid
x <- seq(0,length, length = length_sample)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length_sample**2,2))
field <- rmaxstab(n = 1, coord = grid, cov.mod = model, nugget = 0, range = 1.5, smooth = 1.2)
field_transformed <- array(field, dim = c(length_sample, length_sample))
filled.contour(x, x, field_transformed, color.palette = terrain.colors, nlevel = 20)





length <- 25
x <- seq(0,length, length = length)
grid <- expand.grid(x,x)
grid <- array(unlist(grid), dim = c(length**2,2))
field <- rmaxstab(n = 1, coord = grid, cov.mod = "brown", range = 2, smooth = 1.6)

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


plot(agg, ylim = c(0.5,4))
lines(valid, col = "red")

# Schlather Model
field <- rmaxstab(n = 50, coord = grid, cov.mod = "powexp", nugget = 0, range = 1.5, smooth = 0.8)
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

agg <- aggregate(result[,2], list(round(result[,1],3)), FUN=mean)
names(agg) <- c("h", "Coeff")

#Plot extremal coefficient

schlather_ext_coeff <- function(h, range, smooth){
  corr <- exp(-(h/range)**smooth)
  coeff <- 1+ sqrt((1-corr)/2)
  return(coeff)
}


valid <- data.frame(cbind(agg[,1],schlather_ext_coeff(agg[,1],1.5,0.8)))
names(valid) <- c("h", "Coeff")


plot(agg, ylim = c(0.5,2))
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


