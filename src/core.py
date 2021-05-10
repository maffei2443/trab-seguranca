import re
import math
import random

from utils import xor, slice_lis, ipow
from utils import gcd, iceil

import hashlib as HH
# from utils import hashfyle, binbin2bytes

import utils

from collections import namedtuple



def powers_of_two_divisors(n):
    """Generates iterator from 1 to k such that k is the biggest
        integer satisfying n % (2^k) = 0.
        """
    if n & 1:
        raise ValueError(f"`n` must be even. Got n={n}")
    t = 1
    while not n & 1:
        t += 1
        n >>= 1
    return range(1, t)


# Implementation of pseudocode at [4]
def is_witness(a: int, n: int):
    if not (n & 1) or n < 0:
        raise ValueError(
            f"`n` must be an positive odd. Received n={n}"
        )
    
    t = random.choice(powers_of_two_divisors(n-1))
    u = (n - 1) // ipow(2, t)
    x0 = ipow(a, u, n)
    
    for i in range(1, t+1):
        x1 = ipow(x0, 2, n)
        
        if x1 == 1 and x0 != 1 and x0 != n - 1:
            return True
    
        x0 = x1
    
    if x1 != 1:
        return True
    return False



# Implementation of pseudocode at [4]
def miller_rabin(n: int, s: int):
    if not n >= 2 or not isinstance(n, int):
        raise Exception(
            f"`n` must be of type `int` and >= 2. Got {n}"
        )
    """Miller-Rabin algorithm.
    """
    for _ in range(s):
        a = random.randint(1, n-1)
        if is_witness(a, n):
            return False
            # return 'composite'
    return True
    # return 'prime'


# [7]
def modInverse(a, m):
    m0 = m 
    y = 0
    x = 1
  
    if (m == 1) : 
        return 0
  
    while (a > 1) : 
  
        # q is quotient 
        q = a // m 
  
        t = m 
  
        # m is remainder now, process 
        # same as Euclid's algo 
        m = a % m 
        a = t 
        t = y 
  
        # Update x and y 
        y = x - q * y 
        x = t 
  
  
    # Make x positive 
    if (x < 0) : 
        x = x + m0 
  
    return x 


def gen_n_bits_prime(n: int, s: int = 200):
    mask = (1 << n - 1) | 1

    while True:
        # Check only for odd number AND effectively
        # has `n` bits
        candidate = random.getrandbits(n) | mask
        # print("another try..",)
        if miller_rabin(candidate, s):
            return candidate




def GenModulus(n: int, miller_rabin_tries=200):
    p = gen_n_bits_prime(n, miller_rabin_tries)
    q = gen_n_bits_prime(n, miller_rabin_tries)
    N = p * q
    return (N, p, q)


def GenRSA(n: int, tries=200, dictionary={}):
    N, p, q = GenModulus(n, tries)
    phi_n = (p - 1) * (q - 1)

    e = random.randint(5000, 500_000)
    # e = random.choice(PRIMES)
    while gcd(e, phi_n) > 1:
        e = random.randint(5000, 500_000)
        # print(e, phi_n)
        # input()
    d = modInverse(e, phi_n)


    dictionary['n'] = N
    dictionary['e'] = e
    dictionary['d'] = d
    dictionary['p'] = p
    dictionary['q'] = q

    return N, e, d


