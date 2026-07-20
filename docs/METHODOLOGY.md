# Metodologia implementada

Este documento descreve o comportamento do software. Resultados numéricos de
uma campanha específica pertencem ao relatório científico e não são embutidos
no código nem no conjunto sintético de demonstração.

## 1. Unidade de observação

O banco bruto preserva cada linha válida registrada pelo G-NetTrack. A unidade
usada em mapas analíticos, estatística, exportação científica e painéis é o
**estado analítico deduplicado**.

A chave de deduplicação usa, quando disponíveis:

- timestamp;
- latitude e longitude;
- tecnologia e banda;
- RSRP, RSRQ e SINR.

Antes de remover uma repetição, campos complementares de QoS válidos podem ser
coalescidos dentro do mesmo estado. Valores obtidos por Wi-Fi não são aceitos
como QoS celular. O SQLite bruto permanece inalterado e o número de linhas
retiradas é informado pela API.

## 2. Proveniência e ausências

Campos desconhecidos permanecem nulos e recebem um estado de origem ou falha.
O sistema não usa `fillna(0)` nas análises. As prioridades principais são:

| Variável | Regra efetiva |
|---|---|
| Setor | declaração manual → classificação com buffer → classificação estrita |
| Categoria ambiental | legenda do setor efetivo → buffer → estrita |
| Clima histórico | observação manual → Open-Meteo Archive → ausente |
| Alturas | observação manual → API |
| Contagens e distâncias de objetos | API por ponto → observação manual |

Os campos meteorológicos legados obtidos pelo endpoint `current` são
preservados para auditoria, mas nunca entram em `temperature_c_eff`,
`humidity_eff` ou `cloud_cover_pct_eff`.

O reenriquecimento histórico público é executado por
`scripts/backfill_weather_archive.py`. Ele realiza uma requisição por campanha
ou data, associa o horário mais próximo com tolerância de 90 minutos, preserva
o valor horário e calcula a mediana da campanha. A opção `--dry-run` valida a
consulta e desfaz as alterações.

## 3. Classificação espacial

Os polígonos dos setores são retângulos na referência local do mapa da FEG. A
transformação para WGS84 exige calibração por pontos de controle. Sem
calibração ativa, o sistema não atribui setor automaticamente.

Na classificação complementar são usadas as feições dos setores S01–S21, com
áreas de influência de 15 m para `edificado`, 5 m para `aberto` e 10 m para
`arborizado`, classes a confirmar ou sem classe. Em sobreposição, vence o
polígono com menor distância ao ponto; pontos internos recebem distância
negativa, favorecendo a feição na qual estão mais profundamente inseridos. Um
empate residual preserva a ordem estável dos setores. Registros fora de todos
os buffers permanecem sem classificação.

Categorias manuais especiais, como via e estacionamento, podem ser declaradas
na importação e têm prioridade analítica sobre a classificação automática.

## 4. Sinal e QoS

`signal_rating` é derivado exclusivamente de RSRP:

| Classe | Intervalo |
|---|---|
| Excelente | RSRP > −85 dBm |
| Bom | −95 < RSRP ≤ −85 dBm |
| Satisfatório | −105 < RSRP ≤ −95 dBm |
| Ruim | −115 < RSRP ≤ −105 dBm |
| Péssimo | −125 < RSRP ≤ −115 dBm |
| Nulo | RSRP ≤ −125 dBm ou ausência de conexão válida |

Testes ativos de QoS possuem cadência muito menor que o registro passivo de
rádio. O software não replica uma medida de throughput ou latência para todos
os pontos vizinhos. Linhas marcadas como Wi-Fi mantêm as leituras passivas de
rádio, mas têm QoS celular invalidado.

## 5. Modelos supervisionados

Regressão compara baseline, regressão linear, Random Forest, XGBoost e SVR.
Classificação compara baselines, Random Forest, XGBoost e SVC. Variáveis
categóricas são nominais e passam por codificação ajustada somente no treino de
cada divisão. Imputação e padronização também são ajustadas dentro da divisão.

Métricas de rádio derivadas do alvo são bloqueadas como preditoras para evitar
vazamento. A distância a uma referência estimada dos próprios valores de RSRP
não integra o conjunto padrão de preditores.

A validação agrupada mantém campanhas, setores, datas ou arquivos inteiros em
treino ou teste. A validação aleatória por linha é rotulada como otimista.

## 6. PCA e agrupamentos

As variáveis numéricas são padronizadas. Colunas com pelo menos 50% de ausência
são retiradas e as restantes usam casos completos. O clima só participa pelas
colunas historicamente válidas. PCA, k-means e DBSCAN descrevem perfis
empíricos dentro das rotas; os clusters não são regimes físicos universais.

## 7. Resultados descritivos de propagação

Quando existe uma referência espacial declarada ou estimada, o painel pode
ajustar `RSRP = A − 10 n log10(d)`. O valor exibido é denominado **inclinação
log-distância descritiva**. Ele não é *path loss* absoluto e não substitui um
link budget.

A diferença entre médias outdoor e indoor é apresentada como diferença
descritiva entre grupos. Não é uma medição pareada de perda de penetração O2I.

O projeto não contém recomendação de repetidores ou outra otimização de
implantação. Esse tema permanece somente como possibilidade de trabalho futuro,
dependente de dados de infraestrutura e validação próprios.
