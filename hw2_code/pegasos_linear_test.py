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

X1 = np.append(X,z,axis=1)
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
#    print x.shape
    ywx = np.transpose(y)*np.dot(x, np.transpose(w))
    p2 = 1-ywx[0]
    
    zeros = np.ones((n,))
    val = np.maximum.reduce([zeros, p2])
    return np.sum(val)*1.0/n+reg
#    tot = 0
#    for i in xrange(n):
#        tot+=max(0, 1-y[i]*np.dot(w,x[i]))
#    return tot*1.0/n
#############CODE WORKS, WORK ON PART 2
def pegasos_lambda(x, y, x_init=[None], lr=2**(-8), max_iters=10000):
    iters = 1
    
    # epsilon for convergence criterion
    eps = 1e-6

    # information about the data
    num_samples = x.shape[0]

    # initialize theta (subject to change)
    init_zeros = np.zeros((x.shape[1],))
    theta = init_zeros
   
#    J_err = np.linalg.norm(np.dot(x,theta)-y)**2
    J_err = pegasosErr(x, y, theta, lr)
    while iters < max_iters:
        if iters % 1000 == 0:
            print "Iteration %d" % iters
        if iters == max_iters - 1:
            print "Max iterations (%d iterations) exceeded" % max_iters
            
        x_prime, y_prime = shuffle_in_unison(x,y)
        
        eta = 1.0/(iters*lr)
        for j in xrange(num_samples):
#            print 'hello world'
#            print theta
#            print x_prime[j]
#            print np.dot(np.transpose(theta),x_prime[j])
            if (y_prime[j]*(np.dot(np.transpose(theta),x_prime[j]))<1):
                theta = (1-eta*lr)*theta + eta*y_prime[j]*x_prime[j]
            else:
                theta = (1-eta*lr)*theta
           
#        new_J_err = np.linalg.norm(np.dot(x,theta)-y)**2
        new_J_err = pegasosErr(x,y, theta, lr)
        if abs(new_J_err - J_err) < eps:
            print "Converged after %d iterations with loss %f" % (iters, new_J_err)
            J_err = new_J_err
            break

        J_err = new_J_err
        
        iters += 1
        x_prime, y_prime = shuffle_in_unison(x,y)

    return theta
def pegasos_kern_Err(x, y, alpha, K, lambd):
    reg = lambd/2.0*np.linalg.norm(alpha)**2
    n = x.shape[0]
#    print x.shape
    ywx = np.zeros((n,))
    for j in xrange(n):
#        print 'hello world'
#        print theta
#        print x_prime[j]
#        print np.dot(np.transpose(theta),x_prime[j])
        ywx[i]=y[j]*np.sum((np.dot(np.transpose(alpha),K[j])))
#    ywx = np.transpose(y)*np.sum(np.dot(np.transpose(alpha), K))
    p2 = 1-ywx
#    print ywx
    zeros = np.ones((n,))
    val = np.maximum.reduce([zeros, p2])
    ans = np.sum(val)*1.0/n
#    print ans
    return ans
def pegasos_kernel(x, y, x_init=[None], K=[None], lr = 0.02, max_iters=1000):
    iters = 1
    # epsilon for convergence criterion
    eps = 1e-6

    # information about the data
    num_samples = x.shape[0]

    # initialize theta (subject to change)
    init_zeros = np.ones((x.shape[0],))
    alpha = init_zeros
   
#    J_err = pegasos_kern_Err(x, y, alpha, K, lr)
#    return
    while iters < max_iters:
        if iters % 1000 == 0:
            print "Iteration %d" % iters
        if iters == max_iters - 1:
            print "Max iterations (%d iterations) exceeded" % max_iters
            break
            
        eta = 1.0/(iters*lr)
        for j in xrange(num_samples):
#            print 'hello world'
#            print theta
#            print x_prime[j]
#            print np.dot(np.transpose(theta),x_prime[j])
            if (y[j]*np.sum((np.dot(np.transpose(alpha),K[j])))<1):
                alpha[j] = (1-eta*lr)*alpha[j] + eta*y[j]
            else:
                alpha[j] = (1-eta*lr)*alpha[j]
          
        
        iters += 1
#        x_prime, y_prime = shuffle_in_unison(x,y)

    return alpha


#w = pegasos_lambda(X1,Y)
#print w

def kern(x, y, gamma = 0.2):
    return np.exp(-gamma*np.linalg.norm(x-y)**2)
K = np.zeros((X.shape[0],X.shape[0]))
for i in xrange(X.shape[0]):
    for j in xrange(X.shape[0]):
        K[i][j] = kern(X[i], X[j])
#print K
alpha = pegasos_kernel(X,Y,K=K)
print alpha
# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###
def predict_linearSVM(x):
    if (np.dot(x,w[0:2])+w[2])>0:
        return 1
    return -1


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
#    if np.sum((np.dot(np.transpose(alpha),K)))>1:
#       return 1
#    return -1
#print X.shape[0]

plotDecisionBoundary(X, Y, predict_GaussianSVM, [-1,0,1], title = 'Gaussian SVM')
# plot training results
#plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

