# Segurança e privacidade

## Modelo de implantação

Este projeto é uma ferramenta científica **local**. O servidor deve permanecer
em `127.0.0.1`. Ele não possui autenticação, autorização por usuário, proteção
CSRF ou isolamento multiusuário e não deve ser publicado diretamente na
internet.

## Dados sensíveis

Logs de campo podem conter coordenadas, horários, identificadores de célula,
operadora e nomes de arquivo. Bancos, logs brutos, exportações e `.env` são
ignorados pelo Git. Antes de compartilhar qualquer conjunto, remova ou agregue
esses identificadores e documente a transformação.

Não inclua em issues:

- arquivos `.env` ou credenciais;
- bancos SQLite do projeto;
- logs brutos de coleta;
- URLs de API contendo identificadores;
- coordenadas associadas a pessoas ou rotinas individuais.

## Uploads

O aplicativo aceita apenas TXT, CSV, TSV e LOG, limita o tamanho configurável e
normaliza o nome antes de persistir o arquivo. Essa proteção não substitui um
gateway seguro caso o código seja adaptado para uso em rede.

## Relato de vulnerabilidade

Evite abrir publicamente uma vulnerabilidade que exponha dados. Contate o autor
do repositório por um canal privado disponibilizado no perfil do GitHub e
inclua uma reprodução mínima sem dados reais.
