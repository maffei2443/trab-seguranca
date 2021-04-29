# trab-seguranca


## Decisões de projeto

Para codificação de mensagens, são permitidos os seguintes caracteres:
    ```python
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    ```
}
, além do caractere espaço em branco ` `.
Note-se que esses caracteres possuem código ascii no range [65, 122], sendo portanto representados com 7 bits, a menos do caractere espaço. Para resolver esse incômodo, optei por substituir esses caracteres por `{`, que possui código ascii 123. Esse pré/pós-processamento será transparente ao usuário.







