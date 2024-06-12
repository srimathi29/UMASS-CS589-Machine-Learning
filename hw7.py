# -*- coding: utf-8 -*-
"""hw7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yBCQEwphSdndjOuexGZzcvb-Hvzn2Vkl
"""

import jax.numpy as jnp
from jax import grad
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/drive')
#datapath = '/content/drive/MyDrive/'
datapath = "./"


#################### Task 1 ###################
def plot_results(*args, ylabel, title):
    # Plot the results using Matplotlib
    plt.figure()
    for data, label in args:
        plt.plot(data, label=label)

    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def gradient_descent(x, y, w, alpha, iterations, cost_func, lam=None):
	cost_history = []
	accuracy_history = []
	# print("gradient descent began")
	for i in range(iterations):
		# derivate of least squares is 1/m * (sum model(x,w) -y) x
		gradient = grad(cost_func)(w, x, y, lam)
		w -= alpha * gradient
		cost = cost_func(w, x, y, lam)
		cost_history.append(cost)
		predictions = model(x,w)
		accuracy = calculate_metrics(predictions, y)
		accuracy_history.append(accuracy)
	return w, cost_history, accuracy_history

def calculate_metrics(predictions, labels):
	predicted_classes = np.argmax(predictions, axis = 0)
	return np.mean(predicted_classes == labels)

# multi-class linear classification model
def model(x, w):
	"""
	input:
	- x: shape (N, P)
	- W: shape (N+1, C)

	output:
	- prediction: shape (C, P)
	"""
  # Convert DataFrame to NumPy array if needed
	#if isinstance(x, pd.DataFrame):
		#x = x.to_numpy()
	# option 1: stack 1
	f = x
	#print("just fffff", f)
	#print("before stack 1, x.shape: ", f.shape)

	# tack a 1 onto the top of each input point all at once
	o = jnp.ones((1, np.shape(f)[1]))
	f = jnp.vstack((o,f))
	#x = jnp.vstack([x, jnp.ones(x.shape[1])])
	#print("after stack 1, the X.shape:", f.shape)

	# compute linear combination and return
	a = jnp.dot(f.T,w)

	# option 2:
	#a = w[0, :] + jnp.dot(x.T, w[1:, :])
	return a.T

# standard normalization function
def standard_normalizer(x):
  # compute the mean and standard deviation of the input
  x_means = np.mean(x,axis = 1)[:,np.newaxis]
  x_stds = np.std(x,axis = 1)[:,np.newaxis]

  # check to make sure thta x_stds > small threshold, for those not
  # divide by 1 instead of original standard deviation
  ind = np.argwhere(x_stds < 10**(-2))
  if len(ind) > 0:
      ind = [v[0] for v in ind]
      adjust = np.zeros((x_stds.shape))
      adjust[ind] = 1.0
      x_stds += adjust

 # create standard normalizer function
  normalizer = lambda data: (data - x_means)/x_stds

  # create inverse standard normalizer
  inverse_normalizer = lambda data: data*x_stds + x_means

  # return normalizer
  return normalizer,inverse_normalizer

# compute eigendecomposition of data covariance matrix for PCA transformation
def PCA(x,**kwargs):
  # regularization parameter for numerical stability
  lam = 10**(-7)
  if 'lam' in kwargs:
    am = kwargs['lam']

  # create the correlation matrix
  P = float(x.shape[1])
  Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

  # use numpy function to compute eigenvalues / vectors of correlation matrix
  d,V = np.linalg.eigh(Cov)
  return d,V

# PCA-sphereing - use PCA to normalize input features
def PCA_sphereing(x,**kwargs):
    # Step 1: mean-center the data
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_centered = x - x_means

    # Step 2: compute pca transform on mean-centered data
    d,V = PCA(x_centered,**kwargs)

    # Step 3: divide off standard deviation of each (transformed) input,
    # which are equal to the returned eigenvalues in 'd'.
    stds = (d[:,np.newaxis])**(0.5)

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((stds.shape))
        adjust[ind] = 1.0
        stds += adjust

    normalizer = lambda data: np.dot(V.T,data - x_means)/stds

    # create inverse normalizer
    inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

    # return normalizer
    return normalizer,inverse_normalizer

# multi-class softmax cost function
def multiclass_softmax(w, x_p, y_p, lam=None):
	"""
	Args:
	 	- w: parameters. shape (N+1, C), C= the number of classes
	 	- x_p: input. shape (N, P)
		- y_p: label. shape (1, P)
	Return:
		- softmax cost: shape (1,)
	"""

	# pre-compute predictions on all points
	all_evals = model(x_p,w)
	# print(f"all_evals[:, 0:5].T={all_evals[:, 0:5].T}")

	# logsumexp trick
	maxes = jnp.max(all_evals, axis=0)
	a = maxes + jnp.log(jnp.sum(jnp.exp(all_evals - maxes), axis=0))

	# compute cost in compact form using numpy broadcasting
	b = all_evals[y_p.astype(int).flatten(), jnp.arange(np.size(y_p))]
	cost = jnp.sum(a - b)

	# return average
	return cost/float(np.size(y_p))

def run_task1():

	np.random.seed(42)
	# import MNIST
	x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

	if isinstance(x, pd.DataFrame):
		x = x.values
	if isinstance(y, pd.Series):
		y = y.values

	# re-shape input/output data
	#x = x.values
	x = np.array(x.T)
	y = np.array([int(v) for v in y])

	x = x[:, :50000]  # Ensure x is 784 x 50000
	y = y[:50000][np.newaxis, :]  # Ensure y is 1 x 50000

	print(np.shape(x)) # (784, 70000)
	print(np.shape(y)) # (1, 70000)

	# TODO: fill in your code
  # Initialize parameters
	input_size = x.shape[0]
	num_classes = len(np.unique(y))
	w_init = np.random.randn(input_size + 1, num_classes) * 0.1
	print(np.shape(w_init))

  #After doing experiments
	best_step_length = {}

	best_step_length['original'] =  0.001
	best_step_length['standard'] = 0.1
	best_step_length['pca']      = 0.01

	iterations=10
  # Run gradient descent for each normalization technique
	w_raw, costs_raw, acc_raw = gradient_descent(x, y, w_init, best_step_length['original'], iterations, multiclass_softmax)
	normalizer, _  = standard_normalizer(x)
	x_standard = normalizer(x)
	w_standardized, costs_standardized, acc_standardized = gradient_descent(normalizer(x), y, w_init, best_step_length['standard'], iterations, multiclass_softmax)
	pca_normalizer, _ = PCA_sphereing(x_standard)
	w_pca, costs_pca, acc_pca = gradient_descent(pca_normalizer(x), y, w_init, best_step_length['pca'], iterations, multiclass_softmax)

  # Create plots
	plot_results((costs_raw, 'Raw'), ylabel='Cost', title='Raw Cost')
	plot_results((costs_standardized, 'Standardized'), ylabel='Cost', title='Standardized Cost')
	plot_results((costs_pca, 'PCA'), ylabel='Cost', title='PCA Cost')
	plot_results((costs_raw, 'Raw'), (costs_standardized, 'Standardized'), (costs_pca, 'PCA'), ylabel='Cost', title='All Costs')

	plot_results((acc_raw, 'Raw'), ylabel='Accuracy', title='Raw Accuracy')
	plot_results((acc_standardized, 'Standardized'), ylabel='Accuracy', title='Standardized Accuracy')
	plot_results((acc_pca, 'PCA'), ylabel='Accuracy', title='PCA Accuracy')
	plot_results((acc_raw, 'Raw'), (acc_standardized, 'Standardized'), (acc_pca, 'PCA'), ylabel='Accuracy', title='All Accuracies')


##################
def L1(w, lam):
	return lam * jnp.sum(jnp.abs(w[1:]))

def least_squares(w,x,y,lam):
	cost = jnp.sum((model(x, w) - y)**2) / (2 * jnp.size(y)) + L1(w, lam)/jnp.size(y)
	return cost

def run_task2():
	# load in data
	csvname =  datapath + 'boston_housing.csv'
	data = np.loadtxt(csvname, delimiter = ',')
	x = data[:-1,:]
	y = data[-1:,:]

	print(np.shape(x))
	print(np.shape(y))
	# input shape: (13, 506)
	# output shape: (1, 506)

	# TODO: fill in your code
	normalizer, _ = standard_normalizer(x)
	x_normalized  = normalizer(x)

	#hyperparameters
	alpha = 0.001
	iterations = 5000
	lambdas = [0,50,100,150]

	w_init = jnp.zeros((x_normalized.shape[0]+1,1))

	for lam in lambdas:
		w, cost_history, accuracy_history = gradient_descent(x_normalized, y, w_init, alpha, iterations, least_squares, lam)
		# Plot feature importance
		print(f"Lambda = {lam}")
		print(f"Final cost: {cost_history[-1]}")
		plt.bar(range(len(w)), w.flatten())
		plt.title(f'Lambda = {lam}')
		plt.xlabel('Feature Index')
		plt.ylabel('Weight Value')
		plt.show()

		# Plot cost history
		plt.plot(cost_history)
		plt.title(f'Cost History for Lambda = {lam}')
		plt.xlabel('Iterations')
		plt.ylabel('Cost')
		plt.show()

if __name__ == '__main__':
	run_task1()
	run_task2()