import numpy as np

class Variable(float):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def sample_values(self, samples=1):
        raise NotImplementedError

    def __add__(self, other):
        return self.value.__add__(other)
    
    def __mul__(self, other):
        return self.value.__mul__(other)

    def __repr__(self):
        return str(self.value)


class IntegerVariable(Variable):
    def __init__(self, name, value, min_bound, max_bound):
        super(IntegerVariable, self).__init__(name, value) 
        self.min_bound = min_bound
        self.max_bound = max_bound
    
    def sample_values(self, samples=1):
        return np.random.randint(low = self.min_bound, high = self.max_bound+1, size = samples)


class BinaryVariable(Variable):
    def __init__(self, name, value=None):
        super(BinaryVariable, self).__init__(name, value) 
    
    def sample_values(self, samples=1):
        return np.random.randint(low=0, high=2, size=samples)

class ContinousVariable(Variable):
    def __init__(self, name, value, min_bound, max_bound):
        super(ContinousVariable, self).__init__(name, value) 
        self.min_bound = min_bound
        self.max_bound = max_bound
    
    def sample_values(self, samples=1):
        return np.random.uniform(low=self.min_bound, high=self.max_bound, size=samples)




