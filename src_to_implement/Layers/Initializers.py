import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value # Stores the constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value) # for array creation filled with (value, shape)
    
# Instead of single value, we fill with random values from [0, 1).

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(*weights_shape) # if w.shape = (3,4) then creates a 3x4 NumPy array with random values in [0, 1)
    

class Xavier:
    def initialize(self, dims, fan_in, fan_out):     # Compute standard deviation based on Xavier (Glorot) normal distribution
        scaling = np.sqrt(2 / (fan_in + fan_out))  # This aligns with test expectation
        init_tensor = np.random.normal(loc=0.0, scale=scaling, size=dims)     # Generate values from a normal distribution with mean 0 and calculated std_dev
        return init_tensor
    
class He:
    def initialize(self, shape, fan_in, fan_out):
        scale = (2 / fan_in) ** 0.5       # compute standard deviation (He initialization formula)
        rand = np.random.randn(*shape)    # generate normal-distributed values with mean 0 and std 1
        return rand * scale               # scale the values to the desired standard deviation