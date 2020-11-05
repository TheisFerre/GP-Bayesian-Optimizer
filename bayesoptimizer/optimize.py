import numpy as np
from .gaussianprocess import GP
from .kernels import RBFKernel
from .utils import IntegerVariable, ContinousVariable, BinaryVariable


class BayesianMinimizer(object):
    def __init__(self, func, variables, kernel=None):

        self.func = func
        # Vars X 2 array, where 0 idx is min-bound and 1 idx is max-bound
        self.variables = variables

        if kernel is None:
            self.kernel = RBFKernel(length_scale=1, sigma=1)
        else:
            self.kernel = kernel
        
        self.gp = GP(self.kernel)
    
    def sample_point(self):
        return {variable.name: variable() for variable in self.variables}
    
    def sample_gp_points(self, num_samples):
        points = []
        for variable in self.variables:
            points.append([variable() for _ in range(num_samples)])
        return np.array(points).T

    
    def format_variables(self, variable_dict):
        variable_arr = np.array([variable_dict[variable.name] for variable in self.variables]).reshape(-1, len(self.variables))
        return variable_arr
    
    @staticmethod
    def standardize_variables(array):
        if array.ndim == 2:
            array_mu = np.mean(array, axis=0)
            array_std = np.std(array, axis=0)
        else:
            array_mu = np.mean(array)
            array_std = np.std(array)
        return array_mu, array_std

    def optimize(self, iterations=10, init=None):
        EPS = 1e-8

        if init is None:
            init = self.sample_point()
        else:
            #aqusition function
            pass
        #X_obj = init.reshape(-1, len(self.variables))

        X_obj = self.format_variables(init)
        y_obj = np.array(self.func(**init))

        X_obj_mu, X_obj_std = self.standardize_variables(X_obj)
        y_obj_mu, y_obj_std = self.standardize_variables(y_obj)

        y_obj_mu, y_obj_std = y_obj.mean(), y_obj.std()

        X_obj_stand = (X_obj - X_obj_mu) / (X_obj_std + EPS)
        y_obj_stand = (y_obj - y_obj_mu) / (y_obj_std + EPS)

        self.gp.fit(X_obj_stand, y_obj_stand)

        min_obj_val = y_obj
        min_obj_vars = X_obj

        for _ in range(iterations):
            
            #find next best point
            samples = self.sample_gp_points(num_samples=50)
            samples = (samples - X_obj_mu) / (X_obj_std + EPS)
            mu, _ = self.gp.predict(samples)

            min_sample_idx = np.argmin(mu, axis=0)
            min_sample = samples[min_sample_idx]

            min_sample_transformed = min_sample * (X_obj_std + EPS) + X_obj_mu
            min_sample_input = {variable.name: variable.type(min_sample_transformed[i]) for i, variable in enumerate(self.variables)}
            
            # add point to X_obj
            X_obj = np.append(X_obj, min_sample_transformed.reshape(-1, len(self.variables)), axis=0)

            # compute objective_value
            
            new_obj_val = float(self.func(**min_sample_input))
            #print(f'Objective value: {round(new_obj_val, 4)}, Alpha: {min_sample_transformed}')
            if new_obj_val < min_obj_val:
                min_obj_val = new_obj_val
                min_obj_vars = min_sample_transformed

            y_obj = np.append(y_obj, new_obj_val)


            # refit GP
            X_obj_mu, X_obj_std = self.standardize_variables(X_obj)
            y_obj_mu, y_obj_std = self.standardize_variables(y_obj)

            X_obj_stand = (X_obj - X_obj_mu) / (X_obj_std + EPS)
            y_obj_stand = (y_obj - y_obj_mu) / (y_obj_std + EPS)

            self.gp.fit(X_obj_stand, y_obj_stand)
        
        return min_obj_val, min_obj_vars


    






        

    

