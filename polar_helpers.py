import numpy as np
from numba import njit
import itertools

### GENERAL POLAR CODE ROUTINES ###

@njit(fastmath=True)
def polar_transform(u):
    N = u.size
    n = int(np.log2(N))
    c = u.copy()
    for stage in range(n):
        d = 2**stage
        for i in range(0,N,2*d):
            for j in range(d):
                c[i+j] ^= c[i+j+d]
    return c

def I_min_to_a_vector(I_min, n):
    # Implements Equation (7) from [3]
    # Partial order graph traversal
    def left_swap(i,k):
        if k == 0 and i & 1 == 0:
            return i+1
        elif k > 0 and ((i>>k) & 1)==0 and ((i>>(k-1)) & 1)==1:
            return i + 2**(k-1)
        else:
            return i

    def get_next(i, n):
        x = set()
        if i < 2**n-1:
            for k in range(n):
                lk = left_swap(i,k)
                x.add(lk)
            if i in x: 
                x.remove(i)
        return x
    
    # Make code
    N = 2**n
    a_binary = np.zeros(N, dtype=np.uint8)
    def activate_recursively(i):
        if a_binary[i] == 0:
            a_binary[i] = 1
            for ii in get_next(i, n):
                activate_recursively(ii)
    
    for i in I_min:
        activate_recursively(i)
    
    return a_binary


def bin_matrix_to_longs(A):
    N = A.shape[1]
    return np.sum(2**np.arange(N)*A, axis=1)
    
def longs_to_bin_matrix(l,N):
    ll = l.copy()
    A = np.zeros([len(ll),N], dtype=np.uint8)
    for i in range(N):
        A[:,i] = ll & 1
        ll >>= 1    
    return A


### Affine/Linear Automorphisms ###

def get_stabilizer_block_profile(a_vector):
    # Algorithm 1 from [3]
    
    n = int(np.log2(a_vector.size))
    
    A_set = np.flatnonzero(a_vector)
    A_set_bin = longs_to_bin_matrix(A_set,n)
    s = []    
    i0 = 0
    while i0 < n:
        i1 = n-1
        while i1 >= i0:
            pi = np.arange(n)
            pi[i0] = i1
            pi[i1] = i0
            A_set_p = bin_matrix_to_longs(A_set_bin[:,pi])
            if np.all(a_vector[A_set_p]):
                s.append(i1-i0+1)
                i0 = i1+1
            else:
                i1 = i1-1
    
    return s


def is_invertible(A):
    A = A.copy()
    m = A.shape[0]
    for i in range(m):
        p = np.where(A[i:,i])[0]
        if len(p) == 0:
            return False
        rr = i + p[0]
        row = A[rr,i:].copy()
        A[rr,i:] = A[i,i:]
        A[i,i:] = row
        A[i+1:,i:] ^= A[i+1:,i,None]*row[None,:]
    return True

def get_invertible_matrix(m):
    while True:
        A = np.random.randint(0,2,[m,m])
        if is_invertible(A):
            return A

def random_block_diagonal(block_sizes):
    block_starts = []
    i = 0
    for s in block_sizes:
        block_starts.append(i)
        i += s
    n = i
    A = np.zeros((n,n),dtype=np.uint8)
    for i, size in zip(block_starts, block_sizes):
        j = i+size
        A[i:i+size, i:i+size] = get_invertible_matrix(size)
    return A

def linear_permutation(A):
    # linear map to permutation
    m = A.shape[0]
    z = np.array([list(i) for i in itertools.product([0, 1], repeat=m)])[:,::-1]
    z_prime = (A@z.T)%2
    pi = np.array(bin_matrix_to_longs(z_prime.T),dtype=int)
    return pi

#### References: ####
#
# [3] M. Geiselhart, A. Elkelesh, M. Ebada, S. Cammerer and S. ten Brink,
# "On the Automorphism Group of Polar Codes," 
# 2021 IEEE International Symposium on Information Theory (ISIT), 
# Melbourne, Australia, 2021, pp. 1230-1235, doi: 10.1109/ISIT45174.2021.9518184.