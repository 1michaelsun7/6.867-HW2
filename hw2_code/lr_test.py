from numpy import *
from plotBoundary import *
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as sklin

# import your LR training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = np.array(train[:,0:2])
Y = train[:,2:3]


validate = loadtxt('data/data'+name+'_validate.csv')
XV = validate[:,0:2]
YV = validate[:,2:3]


test = loadtxt('data/data'+name+'_test.csv')
X_test = test[:,0:2]
Y_test = test[:,2:3]

y = Y.ravel()
Y_ad = np.array(y).astype(int)
#print Y_ad.shape
#print X.shape
# Carry out training.
test = np.array([0,1,2])
#print Y.shape
#print X.shape
z = np.ones((X.shape[0],1), dtype=int64)
X1 = np.append(X,z,axis=1)
bs = 0
bw = 0
bb = 0
bp = 0
bl = 0
#for pen in ['l1', 'l2']:
#    for c in [1e9,10000, 5000, 1000,500, 100, 50, 10,1]:
#        lamb = 1.0/c
##        if lamb < 0.01:
##            lamb = 0
#        print "NEXT ITER", pen, lamb
#        logr = sklin.LogisticRegression(penalty = pen, max_iter = 100000, C = c)
#        logr_fit = logr.fit(X,Y_ad)
#        #print logr_fit.get_params()
#        w = logr_fit.coef_[0]
#        print w
##        print w, logr_fit.intercept_[0]
#        b = logr_fit.intercept_[0]
#        print logr_fit.score(X,Y_ad)
#        score = logr_fit.score(XV,YV)
#        print score
#        if score>=bs:
#            bp = pen
#            bl = lamb
#            bs = score
#            bw = w
#            bb = b
#w=bw
#def predictLR_sk(x):
#    if (np.dot(x,w[0:2])+b)>0:
#        return 1
#    return -1
#print bp, bl, bs
#print bw
#print bb
#def acc():
#    tot =0
#    for (x,y) in zip(X_test, Y_test):
#        if predictLR_sk(x)==y:
#            tot+=1
#    return tot*1.0/len(X_test)
#print acc()
#plotDecisionBoundary(X_test, Y_test, predictLR_sk, [0.5], title = "TEST DATASET "+name) ##"dataset" + name + " "+pen+" with lambda = " +str(lamb)

def gradient_error_loss(x, y, w, lambd):
    reg = 2*lambd*w
    ywx = y * (np.dot(x, np.transpose(w)))
    dnll = x* y *(1.0/(1+np.exp(y*np.dot(w,x))))
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
    
def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
    
def stochastic_grad_des(x, y, x_init=[None], lr=1e-5, max_iters=100000):
    lambd = 1
    iters = 0
    # epsilon for convergence criterion
    eps = 1e-3
    # information about the data
    num_samples = x.shape[0]

    # initialize theta (subject to change)
    init_zeros = np.zeros((x.shape[1],))
    
    init_zeros[x.shape[1]-1]=1
#    theta = x_init if x_init.any() else init_zeros
    theta = init_zeros
    theta[1]=1
#    J_err = np.linalg.norm(np.dot(x,theta)-y)**2
    J_err = NLL(x,y,theta)+lambd*np.linalg.norm(theta[:-1])**2
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
        new_J_err = NLL(x,y,theta)+lambd*np.linalg.norm(theta[:-1])**2
        if iters % 200 == 0:
            print abs(new_J_err - J_err)
#            print new_J_err
            print theta
        if abs(new_J_err - J_err) < eps  or np.linalg.norm(grad_J) < eps:
            print "Converged after %d iterations with loss %f" % (iters, new_J_err)
            J_err = new_J_err
            break

        J_err = new_J_err+lambd*np.linalg.norm(theta[:-1])**2
        
        iters += 1
        x_prime, y_prime = shuffle_in_unison(x,y)

    return theta
w = stochastic_grad_des(np.array(X1),np.array(Y))
print w
    
# Define the predictLR(x) function, which uses trained parameters
def predictLR1(x):
    accuracy = 0
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
#print predictLR1(X1)

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
