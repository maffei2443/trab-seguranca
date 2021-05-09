import hashlib as H
import math
import re

"""https://gist.github.com/ppoffice/ 
e10e0a418d5dafdd5efe9495e962d3d2#file-asn1-py"""

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


def hashfyle(p: str):
    return H.sha3_384(open(p,'rb').read()).digest()


from typing import Union

def binstr2int(binAsStr):
    ss = set(binAsStr)
    assert ss.issubset( {'0', '1'} )
    int_val = 0
    for i in binAsStr:
        int_val = 2*int_val + int(i)
    return int_val


def binbytes2int(bb: bytes):
    # codigos ascii de '0' e '1'
    sbb = set(bb)
    assert sbb == {48, 49} or sbb == {0, 1}
    int_val = 0
    def aux(x):
        return 1 if x in (ord('1'), 1) else 0
    
    for i in bb:
        int_val = 2*int_val + aux(i)
    return int_val
    

def bin2int(s: Union[str, bytes]):

    if isinstance(s, str):
        return binstr2int(s)
    elif isinstance(s, bytes):
        return binbytes2int(s)
    raise TypeError("`s` must be instance of `stror `bytes`")
