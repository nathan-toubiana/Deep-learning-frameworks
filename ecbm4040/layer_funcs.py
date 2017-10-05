from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    
    N=x.shape[0]
    D=int(np.prod(x.shape)/N)
    X=np.zeros((N,D))

    for i in range(N):
        X[i]=x[i].reshape(D,)

    out=X.dot(w)+b

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: input data, of shape (N, d_1, ... d_k)
      - w: weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    N=x.shape[0]
    D=int(np.prod(x.shape)/N)
    X=np.zeros((N,D))
    for i in range(N):
        
        X[i]=x[i].reshape(D,)
    
    
    dX = np.dot(dout, w.T)
    dw = np.dot(X.T, dout)
    db = np.dot(dout.T, np.ones(N))
    dx = np.reshape(dX, x.shape)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
   
    out = np.maximum(0, x)
    
   
    return out

def tanh_forward(x):
    """
        Computes the forward pass for rectified linear units (tanhs).
        
        Input:
        - x: inputs, of any shape
        
        Returns a tuple of:
        - out: output, of the same shape as x
        """
  
    out = np.tanh(x)
    
   
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
   
    dx = np.array(dout)
    dx[x <= 0] = 0


    return dx


def tanh_backward(dout, x):
    """
        Computes the backward pass for rectified linear units (tanhs).
        
        Input:
        - dout: upstream derivatives, of any shape
        
        Returns:
        - dx: gradient with respect to x
        """
  
    dx = np.divide(np.array(dout),0.0000000000000000000000000000000001+(1-np.tanh(x)*np.tanh(x)))
    
    
    
    return dx



def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - X: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)
   
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
   

    return loss, dx