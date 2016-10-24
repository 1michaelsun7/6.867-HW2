from numpy import *
from plotBoundary import *
import pylab as pl
import svm_dual as sd

def gaussian_rbf(x1, x2, band):
	return exp(-linalg.norm(x1-x2)**2/(2.0*band**2))

C = [1]
widths = [1]

num=4

for c in C:
	for bw in widths:
		def kf(x1,x2):
			return gaussian_rbf(x1,x2,bw)
		# parameters
		name = str(num)
		print '======Training======'
		# load data from csv files
		train = loadtxt('data/data'+name+'_train.csv')
		# use deep copy here to make cvxopt happy
		X = train[:, 0:2].copy()
		Y = train[:, 2:3].copy()

		GLOBAL_X = X.copy()
		GLOBAL_Y = Y.copy()

		x_len = len(X[0])
		training_dat = [[X[i], Y[i]] for i in xrange(len(Y))]

		# Carry out training, primal and/or dual
		qp_sol = sd.generate_dual_solution(training_dat, c)
		alphas = qp_sol[0][:]

		# w = qp_sol[0][:x_len]
		# b = qp_sol[0][x_len]
		# print(w,b)
		# Define the predictSVM(x) function, which uses trained parameters
		# def predictSVM(x):
		# 	return sign(sum(multiply(alphas, GLOBAL_Y)*vectorize(kf)(GLOBAL_X,x)))
		def predictSVM(x):
			s = 0
			for xi in xrange(len(GLOBAL_X)):
				s += alphas[xi]*GLOBAL_Y[xi]*kf(GLOBAL_X[xi], x)
			return sign(s)

		misclass = 0
		for p in xrange(len(X)):
			if predictSVM(X[p]) != Y[p]:
				misclass += 1

		print("Training Error: ", float(misclass)/len(X))

		# plot training results
		plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


		print '======Validation======'
		# load data from csv files
		validate = loadtxt('data/data'+name+'_test.csv')
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
