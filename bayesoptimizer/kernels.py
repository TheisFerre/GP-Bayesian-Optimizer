import numpy as np

class CorMatrix(np.ndarray):
    def __new__(cls, matrix, kernel):
        self = np.asarray(matrix).view(cls)
        self.kernel = kernel.__class__.__name__
        return self

class RBFKernel(object):
    
    def __init__(self, length_scale, sigma):
        
        self._length_scale = length_scale
        self._sigma = sigma
    
    @property
    def length_scale(self):
        return self._length_scale
    
    @property
    def sigma(self):
        return self._sigma
    
    def __call__(self, X1, X2):
        dist = np.sum(np.square(X1[:,np.newaxis,:] - X2), axis=2)
        matrix = self._sigma**2 * np.exp(-dist / (2*self._length_scale**2))
        cor_matrix = CorMatrix(matrix, self)
        return cor_matrix


class WhiteNoiseKernel(object):
    def __init__(self, sigma):
        self._sigma = sigma

    @property
    def sigma(self):
        return self._sigma
    
    def __call__(self, X1, X2):
        if np.array_equal(X1, X2):
            matrix = np.diag(self._sigma**2 * np.ones(len(X1))  )
            return CorMatrix(matrix, self)
        else:
            return np.zeros((len(X1), len(X2)))





    