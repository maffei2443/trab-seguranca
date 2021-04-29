import re
import math
import random

def ipow(*args):
    return int(pow(*args))


def load_primes(files=['primes/primes1.txt']):
    primes = []
    for fname in files:
        s = open(fname).read()
        s = s.split('edu')[1]
        primes_str = re.findall(r'(\d+)', s)
        primes.extend(map(int, primes_str))
    return primes[500_000:]

PRIMES = load_primes()


def gcd(a, b):
    a, b = max(a,b), min(a,b)
    while b:
        a, b = b, a % b
    return a


def iceil(x):
    return int(math.ceil(x))


def ifloor(x):
    return int(math.floor(x))


def uplog(x):
    return iceil(math.log2(x))


def downlog(x):
    return ifloor(math.log2(x))


def fast_exp(base, e): 
    res = 1 
    while e: 
        if e & 1: 
            res *= base 
        e >>= 1 
        base *= base 
    return res


# [2]
def fast_mod_exp(base, e, mod): 
    res = 1
    base %= mod
    while e: 
        if e & 1: 
            res = (res * base) % mod
        e >>= 1 
        base = (base * base) % mod
    return res


# MathStackExchange
def is_perfect_power(n: int):
    bmax = iceil(math.log2(n))
    for n_root in range(2, bmax+1):
        root = pow(n , 1/n_root)
        down, up = math.floor(root), math.ceil(root)
        if ipow(down, n_root) == n:
            return True
        elif ipow(up, n_root) == n:
            return True
    return False


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


def gt_powers_of_two_divisor(n):
    """Generates iterator from 1 to k such that k is the biggest
        integer satisfying n % (2^k) = 0.
        """

    t = 1
    while not n & 1:
        t += 1
        n >>= 1
    return t


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



        
def get_coprime(supposed_prime, candidates):
    for c in candidates:
        if gcd(c, supposed_prime) == 1:
            return c
    return -1


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
        print(e, phi_n)
        input()
    d = modInverse(e, phi_n)


    dictionary['n'] = N
    dictionary['e'] = e
    dictionary['d'] = d
    dictionary['p'] = p
    dictionary['q'] = q

    return N, e, d


