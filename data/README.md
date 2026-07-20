# Política de dados

Esta pasta é preenchida em tempo de execução e **não contém o banco científico
do projeto no repositório público**.

O aplicativo pode criar:

- `db/`: bancos SQLite locais;
- `raw/`: cópias dos logs importados;
- `exports/`: planilhas exportadas;
- `logs/`: trilhas de execução;
- `calibration.json`, `estimated_site.json` e `site_manual.json`: referências
  geoespaciais locais.

Esses artefatos podem conter coordenadas, horários, identificadores de célula,
operadora e metadados de coleta. Eles são ignorados pelo Git. Não force sua
inclusão com `git add -f`.

Para experimentar o projeto sem dados reais, execute:

```bash
python scripts/generate_demo_log.py
```

O arquivo produzido em `data/demo/` é inteiramente sintético e não deve ser
usado como evidência científica.
