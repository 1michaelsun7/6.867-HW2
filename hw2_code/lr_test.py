from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = np.array(train[:,0:2])
Y = train[:,2:3]

# Carry out training.
### TODO ###
test = np.array([0,1,2])
#print Y.shape
#print X.shape
z = np.ones((X.shape[0],1), dtype=int64)
X1 = np.append(X,z,axis=1)


def gradient_error_loss(x, y, w, lambd):
    reg = 2*lambd*np.linalg.norm(w)
    ywx = y * (np.dot(x, np.transpose(w)))
    dnll = x*y *(1/(1+np.exp(ywx)))
    return dnll - reg 

def NLL(x, y, w):
    tot = 0
    num_samples = x.shape[0]
#    tot = np.sum(np.log(1+ np.exp(y*(np.dot(x,w)))))
    for i in xrange(num_samples):
        val = np.log(1+ np.exp(-y[i]*(np.dot(w,x[i]))))
        tot+=val
    return tot

#print gradient_error_loss(1,2,test,.2)
    
#NLL(X, Y, np.zeros((X.shape[1]+1,)))
def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
    
def stochastic_grad_des(x, y, x_init=[None], lr=0.5e-6, max_iters=10000):
    lambd = 1
    iters = 0
    # epsilon for convergence criterion
    eps = 1e-3
    # information about the data
    num_samples = x.shape[0]

    # initialize theta (subject to change)
    init_zeros = np.zeros((x.shape[1],))
    
    init_zeros[x.shape[1]-1]=0
#    theta = x_init if x_init.any() else init_zeros
    theta = init_zeros
#    J_err = np.linalg.norm(np.dot(x,theta)-y)**2
    J_err = NLL(x,y,theta)
    
    while iters < max_iters:
        if iters % 1000 == 0:
            print "Iteration %d" % iters
        if iters == max_iters - 1:
            print "Max iterations (%d iterations) exceeded" % max_iters

        x_prime, y_prime = shuffle_in_unison(x,y)

        for j in xrange(num_samples):
            
            # d/dTheta - since this is stochastic, we update wrt each data point one at a time
#            grad_J = 1.0/float(num_samples)*(np.dot(x_prime[j],theta)-y_prime[j])*x_prime[j]
            grad_J = gradient_error_loss(x[j],y[j], theta, lambd)
#            delta_t = (lr + iters)**-0.75
#            theta -= delta_t*grad_J
            
            theta = theta+lr*grad_J
            
#        new_J_err = np.linalg.norm(np.dot(x,theta)-y)**2
        new_J_err = NLL(x,y,theta)
        if iters % 100 == 0:
            print abs(new_J_err - J_err)
            print theta
        if abs(new_J_err - J_err) < eps or np.linalg.norm(grad_J) < eps:
            print "Converged after %d iterations with loss %f" % (iters, new_J_err)
            J_err = new_J_err
            break

        J_err = new_J_err
        
        iters += 1
        x_prime, y_prime = shuffle_in_unison(x,y)

    return theta
w = stochastic_grad_des(np.array(X1),np.array(Y))
print w
# Define the predictLR(x) function, which uses trained parameters

#w = [-0.20089726,  0.91568843, -0.3090962 ]
#w= [-0.08410093,  1.65807948, -0.03397636]
def predictLR1(x):
    accuracy = 0
#    w = [-0.18854283, 0.90271942, -0.28683172]
    for j in range(x.shape[0]):
        x1 = x[j]
#        print np.dot(x1,w), Y[j]
        if (np.dot(x1,w))>0:
            if (Y[j]==1):
                accuracy +=1
        else:
            if (Y[j]==-1):
                accuracy+=1
    return accuracy*1.0/x.shape[0]
print predictLR1(X1)

#w = [1.0/i for i in w]
### TODO ###
def predictLR(x):
    if (np.dot(x,w[0:2])+w[2])>0:
        return 1
    return -1

# plot training results


plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
#plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
#pl.show()
