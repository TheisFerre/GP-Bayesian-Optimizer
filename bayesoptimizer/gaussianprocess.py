import numpy as np



class GP(object):
    def __init__(self, kernel, sigma_y=0):

        self.kernel = kernel
        self.fitted = False
    
    def fit(self, X, y):

        self.X = X
        self.y = y
        self.K = self.kernel(X, X)
        self.fitted = True

    def predict(self, X_test, sigma_y=0, return_std=True):
        assert self.fitted, 'Must fit model before predicting!'
        
        K_matrix = self.K + (sigma_y**2 * np.eye(len(self.K)) )
        K_inv = np.linalg.pinv(K_matrix)
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test)

        mu_test = np.array(np.transpose(K_star).dot(K_inv).dot(self.y))

        if return_std:
            std_test = np.array(K_star_star - np.transpose(K_star).dot(K_inv).dot(K_star))
            return np.squeeze(mu_test), np.squeeze(std_test)
        
        return mu_test

