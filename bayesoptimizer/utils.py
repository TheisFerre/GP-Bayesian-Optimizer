import numpy as np

'''class Variable(float):
    def __init__(self, value, name = None):
        self.name = None = name = None
        self.value = value
    
    def sample_values(self, samples=1):
        raise NotImplementedError

    def __add__(self, other):
        return self.value.__add__(other)
    
    def __mul__(self, other):
        return self.value.__mul__(other)

    def __repr__(self):
        return str(self.value)'''

class Variable(float):
    def __new__(cls, value=0, name = None):
        return super().__new__(cls, value)
    def __init__(self, value=0, name = None):
        float.__init__(value)
        self.name = name
    
    def sample_values(self):
        raise NotImplementedError

class IntegerVariable(Variable):
    def __new__(cls, value=0, name=None, min_bound=None, max_bound=None):
        return super().__new__(cls, value, name)
    def __init__(self, value=0, name=None, min_bound=None, max_bound=None):
        super(IntegerVariable, self).__init__(value, name)
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.type = int
    
    def sample_value(self):
        return int(np.random.randint(low = self.min_bound, high = self.max_bound+1))
    
    def __call__(self):
        return self.sample_value()


class BinaryVariable(Variable):
    def __new__(cls, value=0, name=None):
        return super().__new__(cls, value, name)
    def __init__(self, value=0, name=None):
        super(BinaryVariable, self).__init__(value, name)
        self.type = int
    
    def sample_value(self):
        return int(np.random.randint(low=0, high=2))
    
    def __call__(self):
        return self.sample_value()

class ContinousVariable(Variable):
    def __new__(cls, value=0, name=None, min_bound=None, max_bound=None):
        return super().__new__(cls, value, name=None)
    def __init__(self, value=0, name=None, min_bound=None, max_bound=None):
        super(ContinousVariable, self).__init__(value, name) 
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.type = float
    
    def sample_value(self):
        return float(np.random.uniform(low=self.min_bound, high=self.max_bound))
    
    def __call__(self):
        return self.sample_value()




