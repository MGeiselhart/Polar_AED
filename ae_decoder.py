import numpy as np
from numba import njit
import polar_helpers

@njit(fastmath=True)
def _sc_g(x, y, b):
    return (-1)**b * x + y

@njit(fastmath=True)
def _sc_box_plus(x, y):
    return np.sign(x)*np.sign(y) * np.minimum(np.abs(x), np.abs(y)) + np.log(1 + np.exp(-np.abs(x+y))) - np.log(1 + np.exp(-np.abs(x-y)))

@njit(fastmath=True)
def _sc_box_plus_minsum(x, y):
    return np.sign(x)*np.sign(y) * np.minimum(np.abs(x), np.abs(y))

@njit(fastmath=True)
def _sc_decode(llrs, a_vector):
    batch, size = llrs.shape
    if size == 1 and a_vector[0] == 0: # frozen bit
        return np.zeros(llrs.shape,dtype=np.uint8),np.log(1+np.exp(-llrs[:,0]))
    elif np.all(a_vector == 1): # rate 1 node
        return (llrs<0).astype(np.uint8), np.zeros(batch)
    else: # intermediate node
        llr1 = _sc_box_plus(llrs[:,:size//2], llrs[:,size//2:])
        x1, metric1 = _sc_decode(llr1, a_vector[:size//2])
        llr2 = _sc_g(llrs[:,:size//2], llrs[:,size//2:], x1)
        x2, metric2 = _sc_decode(llr2, a_vector[size//2:])
        x = np.hstack((x1 ^ x2, x2))
        metric = metric1 + metric2 
        return x, metric
    
@njit(fastmath=True)
def _sc_decode_minsum(llrs, a_vector):
    batch, size = llrs.shape
    if size == 1 and a_vector[0] == 0: # frozen bit
        return np.zeros(llrs.shape,dtype=np.uint8),np.log(1+np.exp(-llrs[:,0]))
    elif np.all(a_vector == 1): # rate 1 node
        return (llrs<0).astype(np.uint8), np.zeros(batch)
    else: # intermediate node
        llr1 = _sc_box_plus_minsum(llrs[:,:size//2], llrs[:,size//2:])
        x1, metric1 = _sc_decode_minsum(llr1, a_vector[:size//2])
        llr2 = _sc_g(llrs[:,:size//2], llrs[:,size//2:], x1)
        x2, metric2 = _sc_decode_minsum(llr2, a_vector[size//2:])
        x = np.hstack((x1 ^ x2, x2))
        metric = metric1 + metric2 
        return x, metric
    

    
class AE_SC_Decoder:
    # Implements the Path-Metric based AED as proposed in [2]
    
    def __init__(self, a_vector, M, minsum=False):
        self.a_vector = a_vector
        self.N = a_vector.size
        self.decode_func = _sc_decode_minsum if minsum else _sc_decode
        
        s = polar_helpers.get_stabilizer_block_profile(a_vector)
        # make M random permutations from BLTA(s) with b = 0, and below block diagonal = 0
        # This is sufficient due to Thm 2 from [1]
        self.permutations = [polar_helpers.linear_permutation(polar_helpers.random_block_diagonal(s)) for _ in range(M)]
        
    def decode(self, llrs): 
        # apply permutations
        llrs_prime = np.vstack([llrs[pi] for pi in self.permutations])
        
        # SC decode
        c_hat_prime, metrics = self.decode_func(llrs_prime, self.a_vector)
        
        # find best candidate
        best = np.argmin(metrics)
        
        # de-permute
        c_hat = np.empty(self.N, dtype=np.uint8)
        c_hat[self.permutations[best]] = c_hat_prime[best]
        
        # recover message
        u_hat = self.inverse_encode(c_hat)
        return c_hat, u_hat
    
    def inverse_encode(self, c):
        full_u = polar_helpers.polar_transform(c)
        return full_u[np.where(self.a_vector == 1)]
        
#### References: ####
# 
# [1] M. Geiselhart, A. Elkelesh, M. Ebada, S. Cammerer and S. t. Brink, 
# "Automorphism Ensemble Decoding of Reed–Muller Codes,"
# in IEEE Transactions on Communications, vol. 69, no. 10, pp. 6424-6438, Oct. 2021, doi: 10.1109/TCOMM.2021.3098798.
#
# [2] C. Kestel, M. Geiselhart, L. Johannsen, S. ten Brink and N. Wehn, 
# "Automorphism Ensemble Polar Code Decoders for 6G URLLC,"
# WSA & SCC 2023; 26th International ITG Workshop on Smart Antennas and 
# 13th Conference on Systems, Communications, and Coding, Braunschweig, Germany, 2023, pp. 1-6.
