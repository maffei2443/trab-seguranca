#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import hashlib as HH
import random


# In[9]:


from random import getrandbits
import hashlib as HH
from collections import namedtuple

def slice_lis(l:list, b: int):
    lis = []
    for i in range(0, len(l), b):
        lis.append(l[i:i+b])
    return lis

def G(r: bytes):
    return HH.sha3_512(r).digest()


def H(X: bytes):
    return HH.sha3_512(X).digest()


def binstr2binbin(s: str):
    assert set(s) == {'0', '1'}
    b = b''
    for c in s.encode('ascii'):
        if c == ord('1'):
            b += b'1'
        else:
            b += b'0'
    return b


def binbin2bytes(s: bytes):
    return bytes([int(i, 2) for i in slice_lis(s, 8)])


import math
def iceil(x):
    return int(math.ceil(x))


def padded_bin_rep(n: int, pad = 8):
    if n == 0:
        return pad * '0'
    n_rep = bin(n)[2:]
    dt = pad - iceil(math.log2(int(f'0b{n_rep}', 2)))
    return dt*'0' + str(n_rep)



oaep = namedtuple('oaep_enc', 'X Y XY m r m_')
class OAEP:
    # DEPRECATED. Nao seve ser usada, afinal de contas
    """Baseado na descrição da wikipedia"""
    n = 1024
    k0 = 512
    k1 = 128
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
        r = binbin2bytes(
                binstr2binbin(
                        padded_bin_rep(
                            getrandbits(OAEP.k0),
                            OAEP.k0
                        )
                )
        )
        print(type(r), r[:10], '...')
        # G expands the k0 bits of `r` to (n - k0) bits
        X = xor(m_, G(r))
        
        # H reduces the (n-k0) bits of X to k0 bits
        Y = xor(r, H(X))
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
        
        


# In[10]:


def xor(data: bytes, mask: bytes) -> bytes:
    '''Byte-by-byte XOR of two byte arrays'''
    masked = b''
    ldata = len(data)
    lmask = len(mask)
    for i in range(max(ldata, lmask)):
        if i < ldata and i < lmask:
            masked += (data[i] ^ mask[i]).to_bytes(1, byteorder='big')
        elif i < ldata:
            masked += data[i].to_bytes(1, byteorder='big')
        else:
            break
    return masked
