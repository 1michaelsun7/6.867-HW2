from numpy import *
from plotBoundary import *
import pylab as pl
import svm_dual as sd

# def linear_kerfunc(x):


# parameters
name = '4'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

x_len = len(X[0])
print(X[0], Y[0])
training_dat = [[X[i], Y[i]] for i in xrange(len(Y))]
print(len(training_dat))

# Carry out training, primal and/or dual
qp_sol = sd.generate_dual_solution(training_dat, 1)
alphas = qp_sol[0][:]
w = dot(transpose(multiply(alphas, Y)), X)

# w = qp_sol[0][:x_len]
# b = qp_sol[0][x_len]
# print(w,b)
# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
	return sign(dot(squeeze(w), x))

misclass = 0
for p in xrange(len(X)):
	if predictSVM(X[p]) != Y[p]:
		misclass += 1

print("Training Error: ", float(misclass)/len(X))

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

misclass = 0
for p in xrange(len(X)):
	if predictSVM(X[p]) != Y[p]:
		misclass += 1

print("Validation Error: ", float(misclass)/len(X))

# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
