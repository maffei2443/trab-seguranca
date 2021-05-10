# Steps to do

- [x] miller-rabin
- [x] pseudo-random prime generator (miller-rabin test)
- [x] [GenModulus](./img/gen-modulus.png)
- [x] [GenRSA](./img/gen-rsa.png)
- [x] implementar operador `XOR` entre strings
- [x] [OAEP](./img/rsa-oaep.png)
    - [x] Gen
    - [x] Enc
    - [x] Dec

- [] Integrar RSA e OAEP
    - [x] gerar hash de arquivo arbitrário (hashyfile)
    - [x] assinar hash de arquivo usando RSA-OAEP
    - [+-] definir formato para salvar os items: 
      chave módulo, assinatura do hash, conteúdo do arquivo
        - [] faltou salvar a chave do módulo. IDEIA: salvar em arquivo .txt separadamente
