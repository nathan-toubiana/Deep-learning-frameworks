from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
       

        X_feed=X
        for i in range(len(layers)):
            X_feed=layers[i].feedforward(X_feed)
        
        
        loss, dx=softmax_loss(X_feed, y)


        for i in range(len(layers)+1,1,-1):
            dx=layers[i-2].backward(dx)

        
       
        square_weights=0
        for i in range(len(layers)):
            square_weights += np.sum(layers[i].params[0]**2)
        loss += 0.5*self.reg*square_weights
        
       
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
       
        layers=self.layers
        params=layers[0].params
        grads=layers[0].gradients
        for i in range(len(layers)-1):
            params+=layers[i+1].params
            grads+=layers[i+1].gradients
        # Add L2 regularization
        #print(np.array(grads[2]).shape)
        #print(np.array(params[1]).shape)
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        params=[params[i]-learning_rate*grad for i,grad in enumerate(grads)]
       
   
        # update parameters in layers
        for i in range(len(layers)):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
       
        X_feed=X
        for i in range(len(layers)):
            X_feed=layers[i].feedforward(X_feed)


        
        probs = np.exp(X_feed- np.max(X_feed, axis=1, keepdims=True))
        
        
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        predictions=np.argmax(probs,axis=1)

        
        
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


