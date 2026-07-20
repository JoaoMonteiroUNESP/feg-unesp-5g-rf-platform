# Dicionário de dados resumido

O esquema completo está em `app/db.py`. Esta tabela documenta as variáveis mais
usadas na camada analítica.

| Campo | Unidade/tipo | Origem | Uso |
|---|---|---|---|
| `timestamp_log` | data e hora | G-NetTrack | ordem, campanha e validação por data |
| `latitude`, `longitude` | graus WGS84 | GPS do aparelho | mapas, distâncias e setores |
| `gps_accuracy_m` | m | GPS do aparelho | controle de qualidade |
| `rsrp_dbm` | dBm | portadora primária reportada | sinal, alvo de regressão e classes |
| `rsrq_db`, `sinr_db` | dB | portadora primária reportada | caracterização de rádio |
| `frequency_hz` | Hz | derivada do ARFCN | PCA e agrupamentos |
| `network_tech`, `network_mode` | nominal | G-NetTrack | estratificação; sessão NSA não prova portador NR |
| `ping_avg_ms`, `ping_stdev_ms` | ms | teste ativo | QoS, somente conexão móvel válida |
| `test_dl_max_kbps`, `test_ul_max_kbps` | kbit/s | teste ativo | QoS, cadência esparsa |
| `surface_type` | nominal | observação de campo | descrição e modelos supervisionados |
| `indoor_outdoor` | nominal | observação de campo | descrição e diferenças entre grupos |
| `environment_class_effective` | nominal | setor manual/buffer/estrito | descrição e modelos supervisionados |
| `sector_code_effective` | nominal | manual/buffer/estrito | agrupamento espacial e validação |
| `temperature_c_eff` | °C | manual → Archive | modelos, PCA e agrupamentos |
| `humidity_eff` | % | manual → Archive | modelos, PCA e agrupamentos |
| `cloud_cover_pct_eff` | % | manual → Archive | modelos, PCA e agrupamentos |
| `weather_source_eff` | nominal | regra de proveniência | auditoria do clima |
| `building_count_eff` | contagem | Overpass → manual | contexto construído |
| `distance_to_building_m_eff` | m | Overpass → manual | contexto construído |
| `avg_building_height_eff_m` | m | manual → API | contexto construído |
| `tree_count_eff` | contagem | Overpass → manual | vegetação |
| `distance_to_tree_m_eff` | m | Overpass → manual | vegetação |
| `avg_tree_height_eff_m` | m | manual → API | vegetação |
| `tree_density_ndvi` | adimensional | Sentinel-2/GEE | vegetação; opcional |
| `distance_to_serving_m` | m | log ou coordenadas disponíveis | análise exploratória e PCA |
| `distance_to_site_est_m` | m | referência declarada/estimada | descrição relativa; fora do ML padrão |
| `signal_rating` | ordinal derivada | limiares de RSRP | visualização e classificação complementar |

## Grãos disponíveis

- **Bruto:** todas as linhas persistidas, usado em auditoria e exportação
  completa.
- **Analítico:** estados deduplicados em memória, usado por padrão em painéis,
  mapas, estatística e exportação científica.

## Valores que não devem ser combinados

- Clima legado sem sufixo e clima histórico efetivo.
- QoS de conexão Wi-Fi e QoS celular.
- Coordenada estimada de referência e localização oficial de infraestrutura.
- Classe de sinal derivada de RSRP e um alvo independente medido em campo.
