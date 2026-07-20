# Como contribuir

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
```

## Antes de abrir um pull request

```bash
python -m ruff check app tests scripts
python -m pytest -m "not slow"
python -m pytest -m slow
```

Uma contribuição deve:

- incluir teste quando alterar comportamento;
- preservar valores ausentes e a proveniência;
- ajustar transformações apenas dentro dos dados de treino;
- rotular inferências e limitações na interface;
- não adicionar dados de campo, bancos, coordenadas ou credenciais;
- manter o servidor local por padrão.

Mudanças em limiares, regras de deduplicação, buffers espaciais ou features
padrão devem atualizar `docs/METHODOLOGY.md` e os testes correspondentes.
