from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code
import numpy as np

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = np.array(train[:,0:2])
Y = np.array(train[:,2:3])

z = np.ones((X.shape[0],1), dtype=int64)

# Carry out training.
def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
    
def pegasosErr(x,y,w,lambd):
    reg = lambd/2.0*np.linalg.norm(w[:-1])**2
    n = x.shape[0]
    ywx = np.transpose(y)*np.dot(x, np.transpose(w))
    p2 = 1-ywx[0]
    zeros = np.ones((n,))
    val = np.maximum.reduce([zeros, p2])
    return np.sum(val)*1.0/n+reg
    
def fit(x, y, x_init=[None], lr=2**(-10), max_iters=2000):
    iters = 1
    # information about the data
    num_samples = x.shape[0]
    init_zeros = np.zeros((x.shape[1],))
    theta = init_zeros
#    J_err = pegasosErr(x, y, theta, lr)
    while iters < max_iters:
        if iters % 1000 == 0:
            print "Iteration %d" % iters
        if iters == max_iters - 1:
            print "Max iterations (%d iterations) exceeded" % max_iters
            
        x_prime, y_prime = shuffle_in_unison(x,y)
        
        eta = 1.0/(iters*lr)
        for j in xrange(num_samples):
            if (y_prime[j]*(np.dot(np.transpose(theta),x_prime[j]))<1):
                theta = (1-eta*lr)*theta + eta*y_prime[j]*x_prime[j]
            else:
                theta = (1-eta*lr)*theta
#        new_J_err = pegasosErr(x,y, theta, lr)
#        J_err = new_J_err
        iters += 1
        x_prime, y_prime = shuffle_in_unison(x,y)
    return theta
    
# Define the predict_linearSVM(x) function, which uses global trained parameters, w
def predict_linearSVM(x):
    if (np.dot(x,w[0:2])+w[2])>0:
        return 1
    return -1

def plot_Linear():
    # plot training results
    plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM l = 2^-10')
    pl.show()
    
###HOW TO USE PEGASOS LINEAR FIT ----- bias term is added by appending a column of 1's on X
#X1 = np.append(X,z,axis=1)
#w = fit(X1,Y)
#print w
#plot_Linear()
