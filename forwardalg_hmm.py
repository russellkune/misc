import numpy as np 
# forward algorithm for computing HMM with known parameters:

#find probability of x of this form
x = np.array([0,1,2,3,2,1,0,1,2,3])

#Markov Chain S
S = np.zeros((2,2))
S[0,0] = 0.8
S[0,1] = 0.2
S[1,0] = 0.4
S[1,1] = 0.6

#lambda values
Lambda = np.array([1,2])
#stationary dist 
pi = np.array([2.0/3,1.0/3])



# probability of x1 ... xi, S_{i} = k 
# k is either 0 or 1; this is the S state
# S is the transition probs between 0 and 1

def f(x,k,i,S,pi):

	#first we take ith element of x
	x_i = x[(i-1)]
	#lambda value of current k
	l = Lambda[k]
	#emission probability of current x, Poisson Random Variable 

	e_k = l**x_i * np.exp(-l) * (np.math.factorial(x_i))**(-1)

	if i==1:
		return pi[k] * e_k 

	sm = 0
	for j in range(len(S)):

		sm += S[j,k] * f(x,j,i-1,S,pi)

	return e_k * sm


def p(x,S,pi):
	sm = 0
	for j in range(len(S)):
		sm += f(x,j,len(x),S,pi)
	return sm 
