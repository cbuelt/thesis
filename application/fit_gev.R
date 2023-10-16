library(SpatialExtremes)
library(easyNCDF)
library(geosphere)
library(proj4)
library(ncdf4)

current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))




# Define grid
lons <- seq(6.38,7.48,1.1/29)
lats <- seq(50.27,50.97,0.7/29)

lat_lon_grid <- array(unlist(expand.grid(lons,lats)), dim = c(30**2,2))
colnames(lat_lon_grid) <- c("lon", "lat")

grid <- ptransform(lat_lon_grid/180*pi, '+proj=longlat', '+proj=eqc') 
grid[,1] <- grid[,1] - grid[1,1]
grid[,2] <- grid[,2] - grid[1,2]
grid <- grid[,1:2]/4000




data_raw <- nc_open("../../data/application/1931_2020_month_max.nc")
data <- ncvar_get(data_raw, "pr")
n_obs <- dim(data)[3]
data_vec <- as.vector(data)
data_mat <- t(matrix(data_vec, nrow = 900, ncol = n_obs))




loc <- y ~ 1 + lat + lon
scale <- y ~ 1
shape <- y ~ 1
temp_loc <- y ~ year
temp_cov <- cbind(rep(seq(1,90), each = 3))
colnames(temp_cov) <- c("year")
gev_fit <- fitspatgev(data_mat, lat_lon_grid, loc, scale, shape, temp.cov = temp_cov, temp.form.loc = temp_loc)

round(gev_fit$fitted.values,4)
round(gev_fit$std.err,4)
 


