import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import itertools

def generate_primal_solution(inp, C):
	num_points = len(inp)
	w_size = len(inp[0][0])
	print(num_points, w_size)

	P_init = np.zeros((w_size + 1 + num_points, w_size + 1 + num_points)) # +1 for bias
	q_init = np.zeros((w_size + 1 + num_points,))

	P_init[:w_size, :w_size] = np.identity(w_size)
	q_init[-num_points:] = C

	print(P_init, q_init)

	P = matrix(P_init, tc='d')
	q = matrix(q_init, tc='d')

	G_init = np.zeros((2*num_points, w_size + 1 + num_points))
	h_init = np.zeros((2 * num_points,))

	nonzero_constraints = -1*np.identity(num_points)
	nonzero_consts = np.zeros((num_points,))

	G_init[:num_points, -num_points:] = nonzero_constraints
	h_init[:num_points] = nonzero_consts

	hinge_constraints = np.zeros((num_points, w_size + 1 + num_points))
	neg_labels = [-1*dat[1] for dat in inp]
	hinge_constraints[:, :w_size] = np.multiply(np.tile(neg_labels,(1,w_size)), np.array([dat[0] for dat in inp]))
	hinge_constraints[:, w_size] = np.array(np.squeeze(neg_labels))
	hinge_constraints[:, -num_points:] = -1*np.identity(num_points)
	hinge_const = -1*np.ones((num_points,))

	G_init[-num_points:, :] = hinge_constraints
	h_init[-num_points:] = hinge_const

	print(G_init, h_init)

	G = matrix(G_init, tc='d')
	h = matrix(h_init, tc='d')

	sol = solvers.qp(P,q,G,h)
	return sol['x'], sol['primal objective']

def generate_dual_solution(inpt, C):
	num_points = len(inpt)
	print(num_points)

	P_init = np.zeros((num_points,num_points))
	q_init = -1*np.ones((num_points,))

	for i in xrange(num_points):
		for j in xrange(num_points):
			P_init[i][j] = inpt[i][1]*inpt[j][1]*np.dot(inpt[i][0], inpt[j][0])

	print(P_init, q_init)

	P = matrix(P_init, tc='d')
	q = matrix(q_init, tc='d')
	
	G_init = np.zeros((2*num_points, num_points))
	h_init = np.zeros((2*num_points,))

	G_init[:num_points,:] = -1*np.identity(num_points)
	G_init[num_points:, :] = np.identity(num_points)
	h_init[num_points:] = C

	print(G_init, h_init)

	G = matrix(G_init, tc='d')
	h = matrix(h_init, tc='d')

	A_init = np.zeros((1, num_points))
	b_init = np.zeros((1,))

	A_init[0, :] = np.squeeze(np.array([dat[1] for dat in inpt]))

	print(A_init, b_init)

	A = matrix(A_init, tc='d')
	b = matrix(b_init, tc='d')

	sol = solvers.qp(P,q,G,h,A,b)
	return sol['x'], sol['primal objective']

def generate_kernel_solution(inpt, C, kerfunc):
	new_inpt = inpt
	for inp in xrange(len(inpt)):
		x, y = inpt[inp]
		new_inpt[inp] = [kerfunc(x), y]

	return generate_dual_solution(new_inpt, C)

#test_data = [[[2,2], 1], [[2, 3], 1], [[0, -1], -1], [[-3,-2],-1]]
# test_data = [[[-2],-1], [[-.1],-1], [[.1],1], [[1],1]]
# qp_sol = generate_dual_solution(test_data, 10000)
# print(qp_sol[0][:])
# w1,w2,b = qp_sol[0][:3]
# print(w1,w2,b)
# print(qp_sol)

# separator_x = [a/10 for a in xrange(-100,101)]
# separator = [(0.53-0.3*x)/0.46 for x in separator_x]

# plt.plot([2,2], [2,3], 'bo')
# plt.plot([0,-3], [-1,-2], 'ro')
# plt.plot(separator_x,separator)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
