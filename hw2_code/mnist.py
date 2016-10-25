import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from plotBoundary import *
import pylab as pl
import svm_dual as sd

odds = [1,3,5,7,9]
evens = [2,4,6,8,0]
pairwise_classifications = [[1,7],[3,5],[4,9]]
C = [0.01, 0.1, 1, 10, 100]
bands = [0.1,1,4]

def gaussian_rbf(x1, x2, band):
	return exp(-linalg.norm(x1-x2)**2/(2.0*band**2))

def generate_training_set(num):
	name = str(num)

	train = np.genfromtxt('data/mnist_digit_'+name+'.csv', max_rows=200)
	return train

def generate_validation_set(num):
	name = str(num)

	return np.genfromtxt('data/mnist_digit_'+name+'.csv', skip_header=200, max_rows=150)

def generate_test_set(num):
	name = str(num)

	return np.genfromtxt('data/mnist_digit_'+name+'.csv', skip_header=350, max_rows=150)

for pair in pairwise_classifications:
	best_error = 1
	best_C = -float("inf")
	best_w = None
	for c in C:
		neg = pair[0]
		pos = pair[1]
		print '======Training======'
		neg_training = generate_training_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_training_set(n)])
		neg_training = 2*np.true_divide(neg_training, 255) - 1 # normalize

		pos_training = generate_training_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_training_set(p)])
		pos_training = 2*np.true_divide(pos_training, 255) - 1 # normalize

		neg_Y = -1*np.ones((neg_training.shape[0],))
		pos_Y = np.ones((pos_training.shape[0],))

		Xs = np.concatenate((neg_training, pos_training))
		Ys = np.concatenate((neg_Y,pos_Y))
		qp_sol = sd.generate_np_dual_solution(Xs, Ys, c)

		alphas = qp_sol[0][:]
		w = np.sum(np.dot(np.multiply(alphas, Ys), Xs), axis=0)

		print '======Calculating Validation Accuracy======'
		neg_val = generate_validation_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_validation_set(n)])
		neg_val = 2*np.true_divide(neg_val, 255) - 1 # normalize

		pos_val = generate_validation_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_validation_set(p)])
		pos_val = 2*np.true_divide(pos_val, 255) - 1 # normalize

		neg_valY = -1*np.ones((neg_val.shape[0],))
		pos_valY = np.ones((pos_val.shape[0],))

		out_matrix = np.sign(np.dot(w, np.transpose(np.concatenate((neg_val, pos_val)))))
		new_error = 1 - (out_matrix == np.concatenate((neg_valY, pos_valY))).sum()/float(out_matrix.shape[0])
		if new_error < best_error:
			print("Error updating to: ", new_error)
			best_error = new_error
			best_C = c
			best_w = w
	print("Best error: ", best_error)
	print("Best C: ", best_C)

	print '======Testing======'
	neg_test = generate_test_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_test_set(n)])
	neg_test = 2*np.true_divide(neg_test, 255) - 1 # normalize

	pos_test = generate_test_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_test_set(p)])
	pos_test = 2*np.true_divide(pos_test, 255) - 1 # normalize

	neg_testY = -1*np.ones((neg_test.shape[0],))
	pos_testY = np.ones((pos_test.shape[0],))

	test_results = np.sign(np.dot(best_w, np.transpose(np.concatenate((neg_test, pos_test)))))
	accuracy = (test_results == np.concatenate((neg_testY, pos_testY))).sum()/float(test_results.shape[0])
	print("Test Accuracy: ", accuracy)