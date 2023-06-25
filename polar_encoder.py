import numpy as np
from polar_helpers import polar_transform
    
class Polar_Encoder:
    def __init__(self, a_vector):
        self.a_vector = a_vector
        self.N = a_vector.size
        
    def encode(self, u):
        full_u = np.zeros(self.N, np.uint8)
        full_u[np.where(self.a_vector == 1)] = u
        return polar_transform(full_u)
        