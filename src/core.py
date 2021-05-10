import re
import math
import random

from utils import xor, slice_lis, ipow
from utils import gcd, iceil

import hashlib as HH
# from utils import hashfyle, binbin2bytes

import utils

from collections import namedtuple

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
        # print(e, phi_n)
        # input()
    d = modInverse(e, phi_n)


    dictionary['n'] = N
    dictionary['e'] = e
    dictionary['d'] = d
    dictionary['p'] = p
    dictionary['q'] = q

    return N, e, d


def padded_bin_rep(n: int, pad = 8):
    if n == 0:
        return pad * '0'
    n_rep = bin(n)[2:]
    dt = pad - iceil(math.log2(int(f'0b{n_rep}', 2)))
    return dt*'0' + str(n_rep)



# utilizado
def bytes_pow_mod(m, e, n):
    return bytes([pow(i, e, n) for i in m])


oaep = namedtuple('oaep_enc', 'X Y XY m r m_')
class OAEP:
    n = 1024
    k0 = 512
    k1 = 128

    @staticmethod
    def G(r: bytes):
        return HH.sha3_512(r).digest()

    @staticmethod
    def H(X: bytes):
        return HH.sha3_512(X).digest()

    """Baseado na descrição da wikipedia"""
    expected_len_m = (n - k0 - k1) // 8
    
    @staticmethod
    def Enc(m: bytes):
        
        if len(m) != OAEP.expected_len_m:
            raise ValueError(
                f"`m` must by of length {OAEP.expected_len_m}, not {len(m)}"
            )
        # message is padded with k1 zeros to be (n - k0) bits
        m_= m + bytes(OAEP.k1//8)
        
        # r is a randomly generated k0-bit string
        randbits = random.getrandbits(OAEP.k0)

        # r = utils.binbin2bytes(
        #         utils.binstr2binbin(
        #                 padded_bin_rep(
        #                     randbits,
        #                     OAEP.k0
        #                 )
        #         )
        # )
        r = randbits.to_bytes(OAEP.k0 // 8, byteorder='big')
        # print(type(r), r[:10], '...')
        # G expands the k0 bits of `r` to (n - k0) bits
        X = xor(m_, OAEP.G(r))
        
        # H reduces the (n-k0) bits of X to k0 bits
        Y = xor(r, OAEP.H(X))
        return oaep(
            X=X, Y=Y, XY=X+Y, m=m, r=r, m_=m_
        )
    @staticmethod
    def Dec(m: bytes):
        X, Y = slice_lis(m, len(m)//2)
        r = xor(Y, H(X))
        m_ = xor(X, G(r))
        m = m_[:-OAEP.k1//8]
        return oaep(
            X=X, Y=Y, XY=X+Y, m=m, r=r, m_=m_
        )


def SignFile(p: str, ):
    fhash = utils.hashfyle(p)
    fhash_oaep = OAEP.Enc(fhash)
    return fhash_oaep


class RSA_OAEP:
    @staticmethod
    def _dump_signated_file(path, signature):
        with open(f'{path}.signed', 'wb') as fp:
            fp.write( (signature).to_bytes(1024, byteorder='big') )
            fp.write( open(path, 'rb').read() )


    @staticmethod
    def Enc(p: str, rabin_tries=500):
        di = {}
        N, d, e = GenRSA(1024, rabin_tries, di)
        fhash = utils.hashfyle(p)

        fhash_oaep = SignFile(p, )
        
        XY=fhash_oaep.XY
        LXY = list(XY)
        XY_ASSTR=''.join(map(padded_bin_rep, LXY))
        XY_INT=utils.bin2int(XY_ASSTR)
        L = len(XY)   
        sign = pow(XY_INT, di['d'], di['n'])
        message = XY_INT
        return message, sign, di

    def CheckFileSignature(p, e, n):
        with open(p,  'rb') as fp:
            sign = int.from_bytes(fp.read(1024), byteorder='big',)

            _vrfy = pow(sign, e, n)
            
            file_hash=utils.hashfybyte(fp.read())

            file_hash_padded=OAEP.Enc(file_hash)

            if _vrfy == file_hash_padded:
                return True
            else:
                return False


        
    