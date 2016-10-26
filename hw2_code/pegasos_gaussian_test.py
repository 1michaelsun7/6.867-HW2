from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
lmbda = .02;
gamma = 2e-2;

def fit(x, y, K, lr = lmbda, max_iters=1000):
    iters = 1

    # information about the data
    num_samples = x.shape[0]

    init_zeros = np.ones((x.shape[0],))
    alpha = init_zeros
    while iters < max_iters:
        if iters % 1000 == 0:
            print "Iteration %d" % iters
        if iters == max_iters - 1:
            print "Max iterations (%d iterations) exceeded" % max_iters
            break
            
        eta = 1.0/(iters*lr)
        for j in xrange(num_samples):
            if (y[j]*np.sum((np.dot(np.transpose(alpha),K[j])))<1):
                alpha[j] = (1-eta*lr)*alpha[j] + eta*y[j]
            else:
                alpha[j] = (1-eta*lr)*alpha[j]
        iters += 1
    return alpha

def kern(x, y):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)

def predict_GaussianSVM(x):
    tot = 0
    for i in xrange(X.shape[0]):
        if alpha[i]==0:
            continue
        tot+=alpha[i]*kern(x,X[i])
#    tot = kern(x,x)
    if tot>0:
        return 1
    return -1
    
###plotting
def plot_Gaussian():
    plotDecisionBoundary(X, Y, predict_GaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
    pl.show()
    
############HOW TO USE GAUSSIAN TEST FUNC###########################
#K = np.zeros((X.shape[0],X.shape[0]))
#for i in xrange(X.shape[0]):
#    for j in xrange(X.shape[0]):
#        K[i][j] = kern(X[i], X[j])
##print K
###### pass in X, Y, and generated Kernel Matrix
#alpha = fit(X,Y,K=K)
#print alpha
#plot_Gaussian()
