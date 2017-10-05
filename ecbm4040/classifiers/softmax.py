import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    N=(X.shape[0])
    for i in range(N):
        scores = X[i].dot(W)
        scores-=np.max(scores)
        expo=np.exp(scores[y[i]])/np.sum(np.exp(scores))
        loss+=-np.log(expo)
        for j in range(W.shape[1]):
            p=np.exp(scores[j])/np.sum(np.exp(scores))
            dW[:,j]+=(p-(j==y[i]))*X[i]
    

    loss/=N
    dW/=N
    loss += reg * np.sum(W * W)
    dW += reg*2*W
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    N = X.shape[0]
    C = W.shape[1]
            
    scores = X.dot(W)
    scores -= np.matrix(np.max(scores, axis=1)).T

    p=np.exp(scores)/np.matrix(np.sum(np.exp(scores),axis=1)).T
    loss=-np.sum(np.log(p[np.arange(N),y]))
                    
    loss /= N
    loss += reg * np.sum(W * W)
                                            
    
    p[np.arange(N),y] -= 1

    dW += X.T.dot(p)


    dW /= N
    dW += reg*2*W
    

    return loss, dW
