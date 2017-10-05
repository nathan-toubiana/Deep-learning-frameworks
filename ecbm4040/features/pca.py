import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """


    
    X = (X - X.mean()) / X.std()
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    
    U, S, V = np.linalg.svd(cov)
    
    U_reduced = U[:,:K]
    
    T=S/S.sum()
    
    #print(X.shape)
    
    P=U_reduced.T
    #print(P.shape)
    #T=None

   
    
    return (P, T)
