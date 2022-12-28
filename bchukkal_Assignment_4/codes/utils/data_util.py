import math
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import functools
import h5py
from sklearn.model_selection import StratifiedShuffleSplit
import scipy
from scipy.linalg import sqrtm
from sklearn.utils.extmath import svd_flip
import math

#  Generate random numbers between 0 and 1
def generate_random_numbers01(N, dim, max_v = 10000):
    random_ints = np.random.randint(max_v, size=(N, dim))
    init_lb = 0
    return (random_ints - init_lb)/(max_v - 1 - init_lb)

# Generate random numbers between 'lower bound' and 'upper bound'
def generate_random_numbers(N, dim, max_v, lb, ub):
    zero_to_one_points = generate_random_numbers01(N, dim, max_v)
    res = lb + (ub - lb)*zero_to_one_points
    return res

def polynomial_transform(q, X):
    #X = X.reshape(-1, 1)  # Do I need to do this? 
    poly = PolynomialFeatures(q)
    res = poly.fit_transform(X)
    return res


