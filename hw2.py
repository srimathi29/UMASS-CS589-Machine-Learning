# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 01:09:31 2023

@author: Srimathi M
"""

import jax.numpy as jnp
import numpy as np
from jax import grad
# intro of jax library: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html


import matplotlib.pyplot as plt

#################### Task 1 ###################

"""
In this exercise you will implement gradient descent using the hand-computed derivative.
All parts marked "TO DO" are for you to construct.
"""
# plot function
def compare_cost_history(cost_history1, cost_history2, cost_history3):
    plt.figure()
    plt.plot(np.arange(1, len(cost_history1) + 1), cost_history1, 'k-', marker="o", label="Alpha=1")
    plt.plot(np.arange(1, len(cost_history2) + 1), cost_history2, "r-", marker='o', label="Alpha=0.1")
    plt.plot(np.arange(1, len(cost_history3) + 1), cost_history3, 'b-', marker="o", label="Alpha=0.01")

    plt.xlabel("k")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig("comparison.png")


def cost_func(w):
    """
    Params:
    - w (weight)

    Returns:
    - cost (the value of the function)
    """
    cost = (1/50) * (w**4 + w**2 + 10*w)
    return cost

def gradient_func(w):
	"""
	Params:
	- w (weight)

	Returns:
	- grad (gradient of the cost function)
	"""
	gradient = (1/50) * (4*w**3 + 2*w + 10)
	return gradient

def gradient_descent(g, gradient, alpha,max_its,w):
	"""
	Params:
	- g (input function),
	- gradient (gradient function that computes the gradients of the variable)
	- alpha (steplength parameter),
	- max_its (maximum number of iterations),
	- w (initialization)

	Returns:
	- cost_history
	"""
	# run the gradient descent loop
	cost_history = [g(w)]        # container for corresponding cost function history
	for k in range(1,max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value
            grad_eval = gradient_func(w)

        # take gradient descent step
            w = w - alpha*grad_eval

		# collect final weights
            cost_history.append(g(w))

	return cost_history


def run_task1():
	print("run task 1 ...")
	# TODO: Three seperate runs using different steplength

	initial_w = 2.0
	alphas = [1.0, 0.1, 0.01]
	max_iterations = 1000

	cost_history1 = gradient_descent(cost_func, gradient_func, alphas[0], max_iterations, initial_w)
	cost_history2 = gradient_descent(cost_func, gradient_func, alphas[1], max_iterations, initial_w)
	cost_history3 = gradient_descent(cost_func, gradient_func, alphas[2], max_iterations, initial_w)

    #plot cost history
	compare_cost_history(cost_history1, cost_history2, cost_history3)

    #Report the value of the function and its derivative at w
	value_at_w = cost_func(initial_w)
	derivative_at_w = gradient_func(initial_w)
	print(f"Value at w=2 = {value_at_w}, Derivative at w=2 = {derivative_at_w}")
 
	#finding best step length
	print("Final cost using steplength = 1 is", cost_history1[-1])
	print("Final cost using steplength = 0.1 is", cost_history2[-1])
	print("Final cost using steplength = 0.01 is", cost_history3[-1])
 
#print(f"The smallest value is {min_value}")
	print("task 1 finished")


#################### Task 2 ###################

"""
In this exercise you will implement gradient descent
using the automatically computed derivative.
All parts marked "TO DO" are for you to construct.
"""
def cost_history_compare(costs_fixed, costs_diminished):
    plt.figure()
    plt.plot(np.arange(1, len(costs_fixed) + 1), costs_fixed, 'k-', marker="o", label="with fixed steplength")
    plt.plot(np.arange(1, len(costs_diminished) + 1), costs_diminished, "r-", marker='o', label="with diminishing steplength")
    plt.xlabel("k")
    plt.ylabel("cost")
    plt.legend()
    plt.savefig("comparison.png")

def jax_cost_func(w):
    """
    Params:
    - w (weight)

    Returns:
    - cost (the value of the function)
    """
    cost = jnp.abs(w)
    return cost

def gradient_descent_auto(g,alpha,max_its,w, diminishing_alpha=False):
	"""

	gradient descent function using automatic differentiator
	Params:
	- g (input function),
	- alpha (steplength parameter),
	- max_its (maximum number of iterations),
	- w (initialization)

	Returns:
	- weight_history
	- cost_history

	"""
	# TODO: compute gradient module using jax
	grad_f = grad(g)

	# run the gradient descent loop
	weight_history = [w]           # container for weight history
	cost_history = [g(w)]          # container for corresponding cost function history

	for k in range(1, max_its+1):
		# TODO: evaluate the gradient, store current weights and cost function value
		if diminishing_alpha == True:
			alpha = 1/float(k)
		w= w - alpha * grad_f(w)

		#record weight and cost
		weight_history.append(w)
		cost_history.append(g(w))

	return weight_history,cost_history

def run_task2():
	print("run task 2 ...")
	# TODO: implement task 2
	initial_w = 2.0
	max_its = 20

	# Run gradient descent with fixed step length
	fixed_step_length = 0.5
	final_w_fixed, cost_history_fixed = gradient_descent_auto(jax_cost_func, fixed_step_length, max_its, initial_w)

	# Run gradient descent with diminishing step length
	diminishing_step_length = None
	final_w_diminishing, cost_history_diminishing = gradient_descent_auto(jax_cost_func, diminishing_step_length, max_its, initial_w, True)

	cost_history_compare(cost_history_fixed, cost_history_diminishing)

	print("task 2 finished")

if __name__ == '__main__':
	run_task1()
	run_task2()

