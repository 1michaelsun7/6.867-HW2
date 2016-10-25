import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from plotBoundary import *
import pylab as pl
import svm_dual as sd
from sklearn import linear_model as sklin

odds = [1,3,5,7,9]
evens = [2,4,6,8,0]
pairwise_classifications = [[odds, evens]]
C = [0.01, 0.1, 1, 10, 100]
bands = [10**x for x in range(-1, 2)]

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

def LR_MNIST_pair(pair):
	neg = pair[0]
	pos = pair[1]
	print '======Training======'
	neg_training = generate_training_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_training_set(n)])
	#neg_training = 2*np.true_divide(neg_training, 255) - 1 # normalize

	pos_training = generate_training_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_training_set(p)])
	#pos_training = 2*np.true_divide(pos_training, 255) - 1 # normalize

	neg_Y = -1*np.ones((neg_training.shape[0],))
	pos_Y = np.ones((pos_training.shape[0],))

	Xs = np.concatenate((neg_training, pos_training))
	Ys = np.concatenate((neg_Y,pos_Y))

	LR_model = sklin.LogisticRegression()
	LR_model = LR_model.fit(Xs,Ys)
	print("Train: ", LR_model.score(Xs, Ys))

	print '======Calculating Validation Accuracy======'
	neg_val = generate_test_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_validation_set(n)])
	#neg_val = 2*np.true_divide(neg_val, 255) - 1 # normalize

	pos_val = generate_test_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_validation_set(p)])
	#pos_val = 2*np.true_divide(pos_val, 255) - 1 # normalize

	neg_valY = -1*np.ones((neg_val.shape[0],))
	pos_valY = np.ones((pos_val.shape[0],))

	X_val = np.concatenate((neg_val, pos_val))
	Y_val = np.concatenate((neg_valY, pos_valY))

	prediction = LR_model.predict(X_val)
	same = (prediction == Y_val)
	misclassified = np.where(same == 0)
	mis_ex = X_val[misclassified[0][0]]
	print("Misclassified: ", Y_val[misclassified[0][0]])
	pl.imshow(mis_ex.reshape((28,28)), cmap=pl.cm.gray)
	pl.show()

	accuracy = LR_model.score(X_val, Y_val)
	print("Accuracy: ", accuracy)

def CSVM_MNIST_pair(pair):
	best_error = 1
	best_C = -float("inf")
	best_w = None
	best_b = None
	for c in C:
		neg = pair[0]
		pos = pair[1]
		print '======Training======'
		neg_training = generate_training_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_training_set(n)])
		#neg_training = 2*np.true_divide(neg_training, 255) - 1 # normalize

		pos_training = generate_training_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_training_set(p)])
		#pos_training = 2*np.true_divide(pos_training, 255) - 1 # normalize

		neg_Y = -1*np.ones((neg_training.shape[0],))
		pos_Y = np.ones((pos_training.shape[0],))

		Xs = np.concatenate((neg_training, pos_training))
		Ys = np.concatenate((neg_Y,pos_Y))
		qp_sol = sd.generate_np_dual_solution(Xs, Ys, c)

		Ys = np.expand_dims(Ys,1)

		alphas = np.array(qp_sol[0][:])
		supports = np.where(alphas > 1e-8*c)
		num_supports = len(supports[0])

		marginal = np.where((1e-8*c < alphas) & (alphas < c - 1e-8*c))
		num_marginal = len(marginal[0])

		def sum_kf(x):
			return np.sum(np.multiply(alphas[supports], Ys[supports])*np.dot(Xs[supports[0]],x))

		print("Supports: ", num_supports)

		w = np.dot(np.transpose(np.multiply(alphas[supports], Ys[supports])),Xs[supports[0]])
		b = 1.0/num_supports*(np.sum(Ys[supports]-np.apply_along_axis(sum_kf, 1, Xs[marginal[0]])))

		print '======Calculating Validation Accuracy======'
		neg_val = generate_validation_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_validation_set(n)])
		#neg_val = 2*np.true_divide(neg_val, 255) - 1 # normalize

		pos_val = generate_validation_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_validation_set(p)])
		#pos_val = 2*np.true_divide(pos_val, 255) - 1 # normalize

		neg_valY = -1*np.ones((neg_val.shape[0],))
		pos_valY = np.ones((pos_val.shape[0],))

		out_matrix = np.sign(b + np.dot(w, np.transpose(np.concatenate((neg_val, pos_val)))))
		new_error = 1 - (out_matrix == np.concatenate((neg_valY, pos_valY))).sum()/float(out_matrix.shape[0])
		if new_error < best_error:
			print("Error updating to: ", new_error)
			best_error = new_error
			best_C = c
			best_w = w
			best_b = b
	print("Best error: ", best_error)
	print("Best C: ", best_C)

	print '======Testing======'
	neg_test = generate_test_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_test_set(n)])
	#neg_test = 2*np.true_divide(neg_test, 255) - 1 # normalize

	pos_test = generate_test_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_test_set(p)])
	#pos_test = 2*np.true_divide(pos_test, 255) - 1 # normalize

	neg_testY = -1*np.ones((neg_test.shape[0],))
	pos_testY = np.ones((pos_test.shape[0],))

	test_X = np.concatenate((neg_test, pos_test))
	test_Y = np.concatenate((neg_testY, pos_testY))
	test_results = np.sign(best_b + np.dot(best_w, np.transpose(test_X)))
	same = (test_results == test_Y)
	misclassified = np.where(same == 0)
	mis_ex = test_X[misclassified[0][0]]
	print("Misclassified as: ", test_Y[misclassified[0][0]])
	pl.imshow(mis_ex.reshape((28,28)), cmap=pl.cm.gray)
	pl.show()
	accuracy = same.sum()/float(test_results.shape[0])
	print("Test Accuracy: ", accuracy)

def RBF_MNIST_pair(pair):
	best_error = 1
	best_C = -float("inf")
	best_alphas = None
	best_width = None
	best_b = None

	neg = pair[0]
	pos = pair[1]
	# Generate training sets
	neg_training = generate_training_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_training_set(n)])
	neg_training = 2*np.true_divide(neg_training, 255) - 1 # normalize

	pos_training = generate_training_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_training_set(p)])
	pos_training = 2*np.true_divide(pos_training, 255) - 1 # normalize

	neg_Y = -1*np.ones((neg_training.shape[0],))
	pos_Y = np.ones((pos_training.shape[0],))

	Xs = np.concatenate((neg_training, pos_training))
	Ys = np.concatenate((neg_Y,pos_Y))

	# Generate validation sets
	neg_val = generate_validation_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_validation_set(n)])
	neg_val = 2*np.true_divide(neg_val, 255) - 1 # normalize

	pos_val = generate_validation_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_validation_set(p)])
	pos_val = 2*np.true_divide(pos_val, 255) - 1 # normalize

	neg_valY = -1*np.ones((neg_val.shape[0],))
	pos_valY = np.ones((pos_val.shape[0],))

	val_X = np.concatenate((neg_val, pos_val))
	val_Y = np.concatenate((neg_valY, pos_valY))
	for c in C:
		for bw in bands:
			def kf(x1,x2):
				return gaussian_rbf(x1,x2,bw)

			print '======Training======'
			qp_sol = sd.generate_np_kernel_solution(Xs, Ys, c, kf)
			alphas = np.array(qp_sol[0][:])

			support_threshold = 1e-8*c

			supports = np.where(alphas > support_threshold)
			num_supports = len(supports[0])
			print("Supports: ", num_supports)

			marginal = np.where((alphas > support_threshold) & (alphas < c - support_threshold))
			num_marginal = len(marginal[0])

			def sum_kf(x):
				def kern(a):
					return kf(x, a)

				return np.sum(np.multiply(alphas[supports], Ys[supports[0]])*np.apply_along_axis(kern, 1, Xs[supports[0]]))

			b = 1.0/num_marginal*(np.sum(Ys[marginal[0]]-np.apply_along_axis(sum_kf, 1, Xs[marginal[0]])))

			print '======Calculating Validation Accuracy======'
			def predictSVM(x):
				s = 0
				for xi in xrange(len(Xs)):
					s += alphas[xi]*Ys[xi]*kf(Xs[xi], x)
				return sign(s + b)

			val_results = np.squeeze(np.apply_along_axis(predictSVM, 1, val_X))
			val_acc = (val_results == val_Y).sum()/float(val_results.shape[0])
			print("Validation Error: ", 1-val_acc)

			if (1-val_acc) < best_error:
				print("Error updating to: ", 1-val_acc)
				best_error = 1 - val_acc
				best_C = c
				best_width = bw
				best_alphas = alphas
				best_b = b

	print("Best error: ", best_error)
	print("Best C: ", best_C)
	print("Best width: ", best_width)
	def kf(x1,x2):
		return gaussian_rbf(x1,x2,best_width)

	def predictSVM(x):
		s = 0
		for xi in xrange(len(Xs)):
			s += best_alphas[xi]*Ys[xi]*kf(Xs[xi], x)
		return sign(s + best_b)

	print '======Testing======'
	neg_test = generate_test_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_test_set(n)])
	neg_test = 2*np.true_divide(neg_test, 255) - 1 # normalize

	pos_test = generate_test_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_test_set(p)])
	pos_test = 2*np.true_divide(pos_test, 255) - 1 # normalize

	neg_testY = -1*np.ones((neg_test.shape[0],))
	pos_testY = np.ones((pos_test.shape[0],))

	test_X = np.concatenate((neg_test, pos_test))
	test_Y = np.concatenate((neg_testY, pos_testY))

	test_results = np.squeeze(np.apply_along_axis(predictSVM, 1, test_X))
	same = (test_results == test_Y)
	misclassified = np.where(same == 0)
	mis_ex = test_X[misclassified[0][-1]]
	print("Misclassified as: ", test_Y[misclassified[0][-1]])
	pl.imshow(mis_ex.reshape((28,28)), cmap=pl.cm.gray)
	pl.show()
	accuracy = same.sum()/float(test_results.shape[0])
	print("Test Accuracy: ", accuracy)

def PEGASOS_MNIST_pair(pair):
	best_error = 1
	best_C = -float("inf")
	best_alphas = None
	best_width = None
	best_b = None

	neg = pair[0]
	pos = pair[1]
	# Generate training sets
	neg_training = generate_training_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_training_set(n)])
	neg_training = 2*np.true_divide(neg_training, 255) - 1 # normalize

	pos_training = generate_training_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_training_set(p)])
	pos_training = 2*np.true_divide(pos_training, 255) - 1 # normalize

	neg_Y = -1*np.ones((neg_training.shape[0],))
	pos_Y = np.ones((pos_training.shape[0],))

	Xs = np.concatenate((neg_training, pos_training))
	Ys = np.concatenate((neg_Y,pos_Y))

	# Generate validation sets
	neg_val = generate_validation_set(neg) if isinstance(neg, int) else np.array([x for n in neg for x in generate_validation_set(n)])
	neg_val = 2*np.true_divide(neg_val, 255) - 1 # normalize

	pos_val = generate_validation_set(pos) if isinstance(pos, int) else np.array([x for p in pos for x in generate_validation_set(p)])
	pos_val = 2*np.true_divide(pos_val, 255) - 1 # normalize

	neg_valY = -1*np.ones((neg_val.shape[0],))
	pos_valY = np.ones((pos_val.shape[0],))

	val_X = np.concatenate((neg_val, pos_val))
	val_Y = np.concatenate((neg_valY, pos_valY))


for pair in pairwise_classifications:
	LR_MNIST_pair(pair)
	#CSVM_MNIST_pair(pair)
	#RBF_MNIST_pair(pair)