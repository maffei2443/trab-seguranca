import core

class RSA:
    def __init__(self):
        self.di = {}
        self.nde = core.GenRSA(1024, dictionary=self.di)


    def Enc(self, m):
        if isinstance(m, bytes):
            m = int.from_bytes(m, byteorder='big')
        N, d, e = self.nde
        c = pow(m, d, N)
        return c


    def Dec(self, c):
        if isinstance(c, bytes):
            c = int.from_bytes(c, byteorder='big')

        N, d, e = self.nde
        m = pow(c, e, N)
        return m
