#!/usr/bin/env python
# coding: utf-8




import random
import collections
import utils
import hashlib as HH


oaep = collections.namedtuple('oaep_enc', 'X Y XY m r m_')
class OAEP:
    # DEPRECATED. Nao seve ser usada, afinal de contas
    """Baseado na descrição da wikipedia"""
    n = 1024
    k0 = 512
    k1 = 128
    expected_len_m = (n - k0 - k1) // 8

    @staticmethod
    def G(r: bytes):
        return HH.sha3_512(r).digest()


    @staticmethod
    def H(X: bytes):
        return HH.sha3_512(X).digest()


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

        r = randbits.to_bytes(OAEP.k0 // 8, byteorder='big')
        print(type(r), r[:10], '...')
        # G expands the k0 bits of `r` to (n - k0) bits
        X = utils.xor(m_, OAEP.G(r))
        
        # H reduces the (n-k0) bits of X to k0 bits
        Y = utils.xor(r, OAEP.H(X))
        return oaep(
            X=X, Y=Y, XY=X+Y, m=m, r=r, m_=m_
        )


    @staticmethod
    def Dec(m: bytes):
        X, Y = utils.slice_lis(m, len(m)//2)
        r = utils.xor(Y, OAEP.H(X))
        m_ = utils.xor(X, OAEP.G(r))
        m = m_[:-OAEP.k1//8]
        return oaep(
            X=X, Y=Y, XY=X+Y, m=m, r=r, m_=m_
        )
        
