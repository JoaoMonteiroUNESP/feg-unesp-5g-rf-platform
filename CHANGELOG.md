# Changelog

## 0.3.1 — 2026-07-20

- adiciona compatibilidade testada com pandas 3, preservando pandas 2.3 no
  ambiente legado com Python 3.10;
- reconhece `StringDtype` e tipos categóricos nominais antes da codificação
  *one-hot*;
- substitui o uso de `DataFrame._append` nos testes pela API pública
  `pandas.concat`;
- impede que o Dependabot proponha NumPy 2.5+ enquanto a CI permanecer em
  Python 3.11.

## 0.3.0 — 2026-07-19

- prepara o projeto para publicação pública sem dados reais;
- remove a otimização de repetidores do produto;
- torna o grão analítico deduplicado padrão em painéis e estatística;
- separa contagens bruta e analítica;
- corrige interface e documentação para seis classes de sinal;
- restringe o clima analítico a observação manual ou Open-Meteo Archive;
- reclassifica inclinação log-distância e diferença indoor–outdoor como
  resultados descritivos;
- desativa por padrão a referência espacial experimental e rejeita ajustes
  log-distância numericamente implausíveis;
- adiciona proteção de nome, extensão e tamanho em uploads;
- adiciona documentação, CI bloqueante, dados sintéticos e política de
  segurança.

## 0.2.0-p0

- versão interna usada durante o desenvolvimento da Iniciação Científica.
