O trabalho é organizado em 4 módulos lógicos e 5 módulos físicos. Os módulos físicos são:

- [core](src/core.py)
- [utils](utils.py)
- [RSA](src/RSA.py)
- [OAEP][src/OAEP.py]
- [RSA_OAEP](src/RSA_OAEP.py)

Os dois primeiros contém as primitivas matemáticas necessárias para a implementação dos 3 últimos módulos.

### Limitações

- A chave do RSA possui sempre 1024 bits
  - Não que o módulo RSA não permita chaves maiores, mas o módulo **RSA_OAEP** é implementado de modo que isso *é assumido*

