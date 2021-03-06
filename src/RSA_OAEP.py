
import utils
import hashlib as HH

import RSA
import OAEP


class RSA_OAEP:

    dbg = {}
    @staticmethod
    def dump_signated_file(path, signature):
        with open(f'{path}.signed', 'wb') as fp:
            fp.write( signature.to_bytes(1024, byteorder='big') )
            fp.write( open(path, 'rb').read() )


    @staticmethod
    def retrieve_sign_and_hash(path):
        with open(f'{path}.signed', 'rb') as fp:
            sign_bytes = fp.read(1024)
            sign = int.from_bytes(sign_bytes, byteorder='big')
            fhash = HH.sha3_384(fp.read()).digest()
        return sign, fhash


    @staticmethod
    def Sign(p, rsa=None):
        if rsa == None:
            rsa = RSA.RSA()
        # Cria hash dp arquivo
        m = utils.hashfyle(p)
        # Faz padding do hash
        oaep_enc = OAEP.OAEP.Enc(m)
        m_padded = oaep_enc.XY


        

        sign = rsa.Enc(m_padded)

        RSA_OAEP.dbg = rsa.di

        RSA_OAEP.dump_signated_file(p, sign)
        return dict(
            sign=sign,
            n=rsa.di['n'],
            e=rsa.di['e'],
            vrfyer=rsa.Dec
        )


    @staticmethod
    def Vrfy(p_signed, sign, verifier):
        sign, fhash = RSA_OAEP.retrieve_sign_and_hash(p_signed)

        m: int = verifier(sign)
        oaep_dec = OAEP.OAEP.Dec(m.to_bytes(128, byteorder='big'))

        hash_recovered = oaep_dec.m

        return (fhash, hash_recovered)
        
        # vrfy = checker(sign)
        # fhash_recovered = OAEP.OAEP.Dec(vrfy.to_bytes(48, ))
        # return fhash, fhash_recovered


