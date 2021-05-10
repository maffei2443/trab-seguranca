import hashlib as H
import math
import re

from typing import Union

"""https://gist.github.com/ppoffice/ 
e10e0a418d5dafdd5efe9495e962d3d2#file-asn1-py"""

def ipow(*args):
    return int(pow(*args))


def gcd(a, b):
    a, b = max(a,b), min(a,b)
    while b:
        a, b = b, a % b
    return a


def iceil(x):
    return int(math.ceil(x))


"""https://gist.github.com/ppoffice/ 
e10e0a418d5dafdd5efe9495e962d3d2#file-asn1-py"""

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


def slice_lis(l:list, b: int):
    """Retorna fatias de `l` contendo `b` elementos.
    
    A última fatia possui `len(l) % b` elementos.
    """
    lis = []
    for i in range(0, len(l), b):
        lis.append(l[i:i+b])
    return lis




def hashfyle(p: str):
    return H.sha3_384(open(p,'rb').read()).digest()


