import numpy as np
from .gaussianprocess import GP
from .kernels import RBFKernel


class BayesianMinimizer(object):
    def __init__(self, func, bounds, kernel=None):

        self.func = func
        # Vars X 2 array, where 0 idx is min-bound and 1 idx is max-bound
        self.bounds = bounds
        assert bounds.shape[-1] == 2

        if kernel is None:
            self.kernel = RBFKernel(length_scale=1, sigma=1)
        else:
            self.kernel = kernel
        
        self.gp = GP(self.kernel)
    
    def sample_points(self, random=True, num_samples=1):
        if random:
            points = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(num_samples, len(self.bounds)))
        else:
            pass
        return points
    
    @staticmethod
    def standardize_var(X):
        X_mu = X.mean(axis=0)
        X_std = X.std(axis=0)

        return X_mu, X_std

    def optimize(self, iterations=10, init=None):
        EPS = 1e-8

        if init is None:
            init = self.sample_points(num_samples=1)
        else:
            #aqusition function
            pass
        X_obj = init.reshape(-1, len(self.bounds))
        y_obj = np.array(self.func(X_obj))

        X_obj_mu, X_obj_std = self.standardize_var(X_obj)
        y_obj_mu, y_obj_std = y_obj.mean(), y_obj.std()

        X_obj_stand = (X_obj - X_obj_mu) / (X_obj_std + EPS)
        y_obj_stand = (y_obj - y_obj_mu) / (y_obj_std + EPS)

        self.gp.fit(X_obj_stand, y_obj_stand)

        min_obj_val = X_obj
        min_obj_vars = y_obj

        for _ in range(iterations):
            
            #find next best point
            samples = self.sample_points(num_samples=50)
            samples = (samples - X_obj_mu) / (X_obj_std + EPS)
            mu, _ = self.gp.predict(samples)

            min_sample_idx = np.argmin(mu, axis=0)
            min_sample = samples[min_sample_idx]

            min_sample_transformed = min_sample * (X_obj_std + EPS) + X_obj_mu
            # add point to X_obj
            X_obj = np.append(X_obj, min_sample_transformed.reshape(-1, len(self.bounds)), axis=0)
            

            # compute objective_value
            
            new_obj_val = self.func(min_sample_transformed)
            #print(f'Objective value: {round(new_obj_val, 4)}, Alpha: {min_sample_transformed}')
            if new_obj_val < min_obj_val:
                min_obj_val = new_obj_val
                min_obj_vars = min_sample_transformed

            y_obj = np.append(y_obj, new_obj_val)


            # refit GP
            X_obj_mu, X_obj_std = self.standardize_var(X_obj)
            y_obj_mu, y_obj_std = self.standardize_var(y_obj)

            X_obj_stand = (X_obj - X_obj_mu) / (X_obj_std + EPS)
            y_obj_stand = (y_obj - y_obj_mu) / (y_obj_std + EPS)

            self.gp.fit(X_obj_stand, y_obj_stand)
        
        return min_obj_val, min_obj_vars


    






        

    

