from __future__ import print_function

import numpy as np
from ecbm4040.classifiers.linear_svm import *
from ecbm4040.classifiers.softmax import *


class BasicClassifier(object):
    def __init__(self):
        self.W = None
        self.velocity = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, optim='SGD', momentum=0.5, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent(SGD).

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
        - y: a numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - optim: the optimization method, the default optimizer is 'SGD' and
                     feel free to add other optimizers.
        - verbose: (boolean) if true, print progress during optimization.

        Outputs:
        - loss_history: a list containing the value of the loss function of each iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        # Initialize W and velocity(for SGD with momentum)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.W)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            
            #loss, grad = svm_loss_vectorized(self.W, X, y, reg)

            num_batch=num_train//batch_size
            for index in np.random.choice(num_train,(num_batch,batch_size),replace=False):
                X_batch=X[index,:]
                y_batch=y[index]
            
           
                loss,grad=self.loss(X_batch, y_batch, reg)
                self.W=self.W-learning_rate*grad
            loss_history.append(loss)
          

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.

        Returns:
        - y_pred: predicted labels for the data in X. y_pred is a 1-dimensional
                  array of length N, and each element is an integer giving the predicted
                  class.
        """
       
        y_pred=np.argmax(X.dot(self.W),axis=1)

      
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: a numpy array of shape (N, D) containing a minibatch of N
                  data points; each point has dimension D.
        - y_batch: a numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: a tuple containing:
        
        - loss:  a single float
        - gradient:  gradients wrt W, an array of the same shape as W
        """
        
        
        pass


class LinearSVM(BasicClassifier):
    """ A subclass that uses the Multiclass Linear SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(BasicClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
