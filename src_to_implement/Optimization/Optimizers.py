import numpy as np

class Sgd:
    #constructor with a "learning_rate" parameter which should be float
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    #tensor passes as parameter
    def calculate_update(self, weight_tensor: np.ndarray , gradient_tensor: np.ndarray):
        return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity_track = {}  # Tracks momentum for each unique parameter shape

    def calculate_update(self, weight_tensor, gradient_tensor): 
        shape_key = str(np.shape(weight_tensor))  # Use the shape as key to track velocity per unique layer/weight size

        if shape_key not in self.velocity_track: # Initialize velocity if not tracked yet
            self.velocity_track[shape_key] = np.zeros_like(gradient_tensor)
        self.velocity_track[shape_key] = (      # Update velocity using momentum formula: v = μv - η∇L
            self.momentum_rate * self.velocity_track[shape_key]
            - self.learning_rate * gradient_tensor
        )  
        return weight_tensor + self.velocity_track[shape_key] # w = w + v

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu     # beta1
        self.rho = rho   # beta2
        self.step_tracker = {}          # Tracks update step per weight shape
        self.moving_averages = {}       # Stores [m, v] for each shape
        self.epsilon = 1e-8             # Prevents division by zero

    def calculate_update(self, weight_tensor, gradient_tensor):
        shape_key = str(np.shape(weight_tensor)) # for both floats and NumPy arrays.

        if shape_key not in self.moving_averages: # Initialize if this shape hasn't been seen
            self.moving_averages[shape_key] = [np.zeros_like(gradient_tensor),
                                               np.zeros_like(gradient_tensor)]
            self.step_tracker[shape_key] = 0

        m, v = self.moving_averages[shape_key]
        self.step_tracker[shape_key] += 1
        t = self.step_tracker[shape_key]

        m = self.mu * m + (1 - self.mu) * gradient_tensor   # Update moment estimates
        v = self.rho * v + (1 - self.rho) * (gradient_tensor ** 2)

        m_hat = m / (1 - self.mu ** t)      # Bias correction
        v_hat = v / (1 - self.rho ** t)

        self.moving_averages[shape_key] = [m, v]         # updated values strore

        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)    # Compute updated weights