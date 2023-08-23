import numpy as np
from scipy.spatial import distance_matrix


def sample_ll(t, param_grid, grid, data, distmat, cutoff = 5):
    r = param_grid[t,0]
    s = param_grid[t,1]    
    ll = logl(data, np.transpose(data), distmat, r, s)
    res = np.tril(ll, -1)
    res[distmat>=cutoff] = 0

    return np.sum(res), t    


def sample_ll_multiple(t, param_grid, grid, data, distmat, cutoff = 5):
    r = param_grid[t,0]
    s = param_grid[t,1]    
    ll = logl(data, np.transpose(data), np.expand_dims(distmat, axis = 1), r, s)
    res = np.tril(np.mean(ll, axis = 1), -1)
    res[distmat>=cutoff] = 0
    return np.sum(res), t   

def corr_func(h, r, s):
    res = np.exp(-np.power((h/r),s))
    return res    

def s_term(z1, z2, h, r, s):
    res = np.sqrt(1 - 2 * (corr_func(h, r, s)+1) * (z1 * z2)/np.power((z1+z2),2))
    return res

def s_term_dev(z1, z2, h, r, s):
    res = (1/( s_term(z1, z2, h, r, s))) * (-(corr_func(h, r, s) + 1 )) * \
          z2 * (z1+z2) * ((z1+z2)- 2 * z1)/np.power((z1+z2), 4)
    return res

def s_term_dev_2(z1, z2, h, r, s):
    term_1 = (-1/np.power(s_term(z1,z2,h,r,s),2)) * s_term_dev(z2,z1,h,r,s) * \
             (z2/np.power(z1+z2,2) - (2*z1*z2)/np.power(z1+z2,3))
    
    
    term_2 = (1/s_term(z1,z2,h,r,s)) * ((z1-z2)/np.power(z1+z2,3) - (2*z1*z1-4*z1*z2)/np.power(z1+z2,4))
    
    res = -(corr_func(h,r,s)+1)*(term_1+term_2)
    return res

def V(z1, z2, h, r, s):
    res = 0.5 * (1/z1 + 1/z2) * (1 + s_term(z1, z2, h, r, s))
    return res

def V1(z1, z2, h, r, s):
    res = 0.5 * (-1/np.power(z1, 2)*(1+s_term(z1, z2, h, r, s)) + \
                 (1/z1 + 1/z2)*s_term_dev(z1,z2,h,r,s))
    return res

def V2(z1, z2, h, r, s):
    res = V1(z2, z1, h, r, s)
    return res

def V12(z1, z2, h, r, s):
    res = 0.5 * ((-1/np.power(z1,2)) * s_term_dev(z2, z1, h, r, s) + \
          (-1/np.power(z2,2)) * s_term_dev(z1, z2, h, r, s) + (1/z1 + 1/z2) * s_term_dev_2(z1, z2, h, r, s))
    return res

def logl(z1, z2, h, r, s):
    res = np.log(V1(z1, z2, h, r, s) * V2(z1, z2, h, r, s) - V12(z1, z2, h, r, s)) - V(z1, z2, h, r, s)
    return res

