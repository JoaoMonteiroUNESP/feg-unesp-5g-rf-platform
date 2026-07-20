"""
FastAPI entrypoint. Serves the auditing-aware dashboard and mounts the API
router. Run with:

    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app import __version__
from app.config import settings
from app.db import init_db
from app.api import router
from app.logging_config import configure_logging, get_logger


configure_logging()
log = get_logger("feg")


app = FastAPI(
    title="FEG-UNESP — RF Research Platform",
    description="Plataforma científica para análise de cobertura 4G/5G "
                "no campus FEG-UNESP. Coleta real, parser auditável, "
                "estatística rigorosa e modelos supervisionados com "
                "validação cruzada e matriz de confusão.",
    version=__version__,
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    url = f"http://{settings.host}:{settings.port}"
    bar = "═" * 64
    msg = (
        f"\n{bar}\n"
        f"  FEG-UNESP RF Research Platform\n"
        f"  Dashboard:    {url}\n"
        f"  API health:   {url}/api/health\n"
        f"  API summary:  {url}/api/summary\n"
        f"  DB:           {settings.db_url}\n"
        f"  Logs (JSON):  data/logs/feg.log\n"
        f"{bar}\n"
    )
    log.info(msg)


app.include_router(router)


HTML = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "templates" / "index.html").exists() else None


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD)


# ---------------------------------------------------------------------------
# Dashboard (single-file HTML + Leaflet + leaflet.heat).
# Visual style: clean scientific neutral tones, JetBrains Mono for numbers.
# ---------------------------------------------------------------------------
_DASHBOARD = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>FEG-UNESP — Plataforma Científica RF</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
:root{
  --bg:#0E1424; --card:#152034; --ink:#E8ECF3; --muted:#94A3B8;
  --accent:#0EA5A4; --warn:#F59E0B; --danger:#E11D48; --good:#22C55E;
  --line:#1F2A44;
}
*{box-sizing:border-box;}
html,body{height:100%;margin:0;background:var(--bg);color:var(--ink);font-family:Inter,sans-serif;}
.num,code,pre,kbd{font-family:'JetBrains Mono',monospace;}
header{background:#0A1020;border-bottom:1px solid var(--line);padding:10px 18px;display:flex;justify-content:space-between;align-items:center;}
header .brand{font-weight:800;letter-spacing:.5px;}
header .brand span{color:var(--accent);}
header .meta{font-size:.72rem;color:var(--muted);text-align:right;line-height:1.4;}
.layout{display:grid;grid-template-columns:380px 1fr;gap:14px;padding:14px;height:calc(100vh - 56px);}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;position:relative;}
.panel section[data-pane]{min-height:0;}
/* Containers cujo conteúdo gerado pode ser mais largo que o painel — eles
   ganham scroll horizontal próprio sem afetar o resto. */
#stats_tables, #cls_summary, #reg_charts, #unsup_summary, #ingest_dump, #runs_list{
  overflow-x:auto; max-width:100%;
}
#stats_tables table, #cls_summary table, #runs_list table, #ingest_dump table{
  width:max-content; min-width:100%;
}
details.json-fold{margin-top:8px;}
details.json-fold > summary{cursor:pointer;font-size:.66rem;color:var(--muted);letter-spacing:.6px;text-transform:uppercase;padding:4px 0;}
details.json-fold[open] > summary{color:var(--accent);}
/* Botão expandir/recolher painel */
.panel-toggle{position:absolute;top:8px;right:8px;background:#0F182A;border:1px solid var(--line);color:var(--muted);width:26px;height:26px;border-radius:6px;cursor:pointer;font-size:.85rem;line-height:1;font-weight:700;}
.panel-toggle:hover{border-color:var(--accent);color:var(--accent);}
.layout.expanded{grid-template-columns:1fr 0 !important;}
.layout.expanded > .panel:nth-child(2){display:none;}
.panel h2{font-size:.78rem;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin:0 0 10px 0;}
.tabs{display:flex;gap:4px;margin-bottom:10px;flex-wrap:wrap;}
.tab{flex:1 1 auto;background:#0F182A;border:1px solid var(--line);color:var(--muted);font-weight:700;font-size:.70rem;padding:8px 6px;border-radius:6px;cursor:pointer;text-transform:uppercase;letter-spacing:.5px;}
.tab.active{background:var(--accent);border-color:var(--accent);color:#0A1020;}
.btn{display:inline-flex;justify-content:center;align-items:center;gap:6px;width:100%;padding:9px 10px;border-radius:6px;border:1px solid var(--line);background:#0F182A;color:var(--ink);font-weight:700;font-size:.7rem;cursor:pointer;letter-spacing:.4px;}
.btn:hover{border-color:var(--accent);}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#0A1020;}
.btn.danger{background:var(--danger);border-color:var(--danger);color:#fff;}
.field{display:block;margin:6px 0;font-size:.7rem;color:var(--muted);}
input[type="file"],input[type="number"],input[type="text"],select{width:100%;background:#0A1020;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-size:.78rem;font-family:'JetBrains Mono',monospace;}
.row{display:flex;gap:8px;}
.row > *{flex:1;}
.kvs{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:.72rem;}
.kvs .k{color:var(--muted);}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:8px;margin:10px 0;}
.kpi-card{background:#0F182A;border:1px solid var(--line);border-radius:8px;padding:10px 12px;}
.kpi-card .lbl{font-size:.62rem;letter-spacing:.7px;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}
.kpi-card .val{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;color:var(--ink);line-height:1.1;}
.kpi-card .sub{font-size:.65rem;color:var(--muted);margin-top:2px;}
.kpi-card.good .val{color:var(--good);}
.kpi-card.warn .val{color:var(--warn);}
.kpi-card.bad .val{color:var(--danger);}
.dash-section{margin-top:14px;}
.dash-section h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:var(--accent);margin:8px 0 6px 0;}
.dash-plot{background:#0A1020;border:1px solid var(--line);border-radius:6px;padding:6px;margin-bottom:8px;}
.dash-table{width:100%;border-collapse:collapse;font-size:.72rem;margin:4px 0 10px 0;}
.dash-table th{background:#0F182A;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-size:.62rem;padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;}
.dash-table td{padding:5px 8px;border-bottom:1px solid var(--line);font-family:'JetBrains Mono',monospace;}
.dash-table tr:hover td{background:#0F182A;}
.dash-bar{height:6px;background:var(--accent);border-radius:3px;display:inline-block;}
.warn{background:#241B0A;color:#FCD34D;border-left:3px solid var(--warn);padding:6px 8px;font-size:.7rem;border-radius:4px;margin-top:6px;white-space:pre-wrap;}
.ok{color:var(--good);font-weight:700;}
.bad{color:var(--danger);font-weight:700;}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;background:#0F182A;border:1px solid var(--line);color:var(--muted);font-size:.65rem;letter-spacing:.4px;}
pre.out{flex:1;background:#0A1020;border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font-size:.7rem;line-height:1.45;color:#CBD5E1;white-space:pre-wrap;margin:0;}
.map-shell{position:relative;height:100%;border-radius:10px;overflow:hidden;border:1px solid var(--line);}
#map{height:100%;width:100%;}
.legend{position:absolute;bottom:10px;right:10px;background:rgba(15,24,42,.92);border:1px solid var(--line);padding:8px 10px;border-radius:6px;font-size:.7rem;line-height:1.5;}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.controls{position:absolute;top:10px;right:10px;background:rgba(15,24,42,.92);border:1px solid var(--line);padding:8px 10px;border-radius:6px;font-size:.7rem;display:flex;gap:8px;align-items:center;}
.controls select{width:auto;padding:4px 6px;font-size:.7rem;}
.controls .k{color:var(--muted);}
.scroll{overflow-y:auto;flex:1;}
table{width:100%;border-collapse:collapse;font-size:.7rem;}
th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;}
th{color:var(--muted);text-transform:uppercase;letter-spacing:.6px;font-size:.62rem;}
.cm{display:grid;grid-auto-flow:row;gap:2px;font-size:.7rem;}
.cm .cell{padding:6px 8px;border-radius:3px;background:#0F182A;text-align:center;font-family:'JetBrains Mono',monospace;}
</style>
</head>
<body>
<header>
  <div class="brand">FEG <span>UNESP</span> — Plataforma Científica RF</div>
  <div style="display:flex;align-items:center;gap:14px;">
    <div class="meta" id="hdr_summary">carregando estado…</div>
    <div style="display:flex;flex-direction:column;gap:4px;">
      <button class="btn primary" id="hdrExportSci" style="width:auto;padding:5px 10px;font-size:.65rem;">📥 Exportar científico (Quadro 2)</button>
      <button class="btn" id="hdrExportFull" style="width:auto;padding:5px 10px;font-size:.65rem;">📥 Exportar completo</button>
    </div>
  </div>
</header>

<div id="cal_banner" style="display:none;background:#3F1D0A;color:#FCD34D;border-bottom:1px solid #F59E0B;padding:8px 18px;font-size:.78rem;letter-spacing:.3px;">
</div>
<div class="layout">
  <!-- LEFT PANEL -->
  <div class="panel">
    <button class="panel-toggle" id="btnExpand" title="Expandir painel (oculta o mapa)">⛶</button>
    <div class="tabs">
      <button class="tab active" data-tab="ingest">INGESTÃO</button>
      <button class="tab" data-tab="dashboard">DASHBOARD</button>
      <button class="tab" data-tab="stats">ESTATÍSTICA</button>
      <button class="tab" data-tab="reg">ML · REGRESSÃO</button>
      <button class="tab" data-tab="cls">ML · CLASSIFICAÇÃO</button>
      <button class="tab" data-tab="unsup">ML · PCA/CLUSTERS</button>
      <button class="tab" data-tab="audit">AUDITORIA</button>
      <button class="tab" data-tab="calib">CALIBRAÇÃO</button>
    </div>

    <!-- TAB: INGEST -->
    <section data-pane="ingest" style="display:block;">
      <h2>Upload do log G-NetTrack</h2>
      <p class="field">Aceita logs CSV/TSV/TXT — formato detectado automaticamente
         (gnettrack_full ou gnettrack_cellfind). Sem mocks. Variáveis ausentes
         são marcadas, nunca inventadas.</p>
      <input type="file" id="logIn" accept=".txt,.csv,.tsv,.log" multiple>
      <label class="field">Campaign ID (opcional, mas RECOMENDADO)</label>
      <input type="text" id="campaignIn"
             placeholder="ex: manha-2026-05-04, uti-pico-18h">
      <label class="field">Localização do quadrante (Quadro 2)</label>
      <select id="ioSel">
        <option value="">— não anotar —</option>
        <option value="outdoor">outdoor</option>
        <option value="indoor">indoor</option>
      </select>
      <label class="field" title="Quando informado, tem prioridade sobre a classificação automática nas análises por setor.">Setor declarado (PRIORITÁRIO sobre o automático)</label>
      <select id="manualSectorSel">
        <option value="">— não informar (usa classificação por buffer) —</option>
      </select>
      <label class="field" title="Tipo de superfície predominante no local da coleta. Afeta reflexão do sinal.">Tipo de superfície</label>
      <select id="surfaceSel">
        <option value="">— não informar —</option>
        <option value="grama">grama</option>
        <option value="terra">terra</option>
        <option value="asfalto">asfalto</option>
        <option value="concreto">concreto</option>
        <option value="misto">misto</option>
      </select>
      <div class="row">
        <div>
          <label class="field" title="Altura média estimada dos prédios no entorno (em metros). Preenche a lacuna do OSM.">Altura média de prédios (m)</label>
          <input type="number" id="bldHeightIn" min="0" max="100" step="0.5" placeholder="ex: 10.5">
        </div>
        <div>
          <label class="field" title="Altura média estimada das árvores no entorno (em metros). Preenche a lacuna do OSM.">Altura média de árvores (m)</label>
          <input type="number" id="treeHeightIn" min="0" max="40" step="0.5" placeholder="ex: 8">
        </div>
      </div>
      <div class="row">
        <div>
          <label class="field">Quantidade de prédios próximos</label>
          <input type="number" id="bldCountIn" min="0" max="50" step="1" placeholder="ex: 3">
        </div>
        <div>
          <label class="field">Distância ao prédio mais próximo (m)</label>
          <input type="number" id="bldDistIn" min="0" max="500" step="0.5" placeholder="ex: 7">
        </div>
      </div>
      <div class="row">
        <div>
          <label class="field">Quantidade de árvores próximas</label>
          <input type="number" id="treeCountIn" min="0" max="100" step="1" placeholder="ex: 4">
        </div>
        <div>
          <label class="field">Distância à árvore mais próxima (m)</label>
          <input type="number" id="treeDistIn" min="0" max="500" step="0.5" placeholder="ex: 3.5">
        </div>
      </div>
      <div class="row">
        <div>
          <label class="field" title="Temperatura anotada em campo. Dados históricos de API só são válidos quando alinhados ao período da campanha.">Temperatura (°C)</label>
          <input type="number" id="tempIn" min="-10" max="60" step="0.1" placeholder="ex: 24">
        </div>
        <div>
          <label class="field" title="Umidade anotada em campo. O endpoint meteorológico atual não substitui observação histórica.">Umidade (%)</label>
          <input type="number" id="humIn" min="0" max="100" step="1" placeholder="ex: 60">
        </div>
      </div>
      <label class="field" title="Cobertura de nuvens (0 = céu limpo · 100 = totalmente nublado).">Cobertura de nuvens (%)</label>
      <input type="number" id="cloudIn" min="0" max="100" step="5" placeholder="ex: 25">
      <label class="field" title="0=nenhuma vegetação · 1=esparsa · 2=média · 3=densa. Fallback se Earth Engine/NDVI estiver offline.">Densidade visual de vegetação</label>
      <select id="vegDensSel">
        <option value="">— não informar (usa NDVI da API se disponível) —</option>
        <option value="0">0 — nenhuma</option>
        <option value="1">1 — esparsa</option>
        <option value="2">2 — média</option>
        <option value="3">3 — densa</option>
      </select>
      <label class="field" title="Situação de chuva no momento da coleta.">Situação de chuva</label>
      <select id="precipSel">
        <option value="">— não informar —</option>
        <option value="0">0 — seco</option>
        <option value="1">1 — garoa</option>
        <option value="2">2 — chuva leve</option>
        <option value="3">3 — chuva moderada</option>
        <option value="4">4 — chuva forte</option>
      </select>
      <label class="field" title="O quanto o céu está obstruído acima de você. Aproxima LoS (linha de visada) vs NLoS.">Obstrução visual do céu</label>
      <select id="obsSel">
        <option value="">— não informar —</option>
        <option value="0">0 — livre (LoS)</option>
        <option value="1">1 — parcial</option>
        <option value="2">2 — bloqueada (NLoS)</option>
      </select>
      <label class="field"><input type="checkbox" id="enrich" checked>
        Enriquecer via APIs externas (clima atual apenas para auditoria; Overpass / GEE)
      </label>
      <label class="field"><input type="checkbox" id="forceUp">
        Forçar reingestão mesmo se SHA já existir
      </label>
      <button class="btn primary" id="btnUpload">Enviar e processar</button>
      <button class="btn" id="btnEnrichExisting" style="margin-top:6px;">Enriquecer medições já ingeridas</button>
      <button class="btn" id="btnCampaigns" style="margin-top:6px;">Listar campanhas</button>
      <button class="btn" id="btnTempCov" style="margin-top:6px;">Cobertura temporal por setor</button>
      <div id="ingest_out" class="warn" style="display:none"></div>
      <pre class="out" id="ingest_dump">Nenhum upload nesta sessão.</pre>
    </section>

    <!-- TAB: DASHBOARD -->
    <section data-pane="dashboard" style="display:none;">
      <h2>Painel de resultados</h2>
      <div class="row" style="gap:6px;">
        <select id="dashCampaign" style="flex:2;">
          <option value="">— todas as campanhas —</option>
        </select>
        <button class="btn primary" id="btnDashRefresh" style="flex:1;">Atualizar</button>
      </div>
      <div id="dash_loading" class="field" style="margin-top:8px;display:none;color:var(--accent);">
        Carregando…
      </div>
      <div id="dash_kpis" class="kpi-grid"></div>
      <div id="dash_charts"></div>
      <div id="dash_tables"></div>
    </section>

    <!-- TAB: STATS -->
    <section data-pane="stats" style="display:none;">
      <h2>Estatística (ANOVA + FDR + sumário)</h2>
      <label class="field">Fator categórico (ANOVA)</label>
      <select id="anovaFactor">
        <option value="network_tech">network_tech (5G vs 4G)</option>
        <option value="signal_rating">signal_rating</option>
        <option value="environment_class">environment_class</option>
        <option value="sector_code">sector_code</option>
        <option value="campaign_id">campaign_id (horário/sessão)</option>
      </select>
      <label class="field">Resposta contínua</label>
      <select id="anovaResponse">
        <option value="rsrp_dbm">rsrp_dbm</option>
        <option value="sinr_db">sinr_db</option>
        <option value="ping_avg_ms">ping_avg_ms (latência)</option>
        <option value="ping_stdev_ms">ping_stdev_ms (jitter)</option>
        <option value="test_dl_max_kbps">test_dl_max_kbps</option>
      </select>
      <label class="field"><input type="checkbox" id="statsRobust">
        ANOVA robusta (Shapiro + Levene + Welch + Kruskal-Wallis + ω² + Hedges' g)
      </label>
      <button class="btn primary" id="btnStats">Rodar ANOVA + Pearson(FDR)</button>
      <div class="row" style="margin-top:6px;">
        <button class="btn" id="btnBySector">Resumo por setor</button>
        <button class="btn" id="btnByEnv">Resumo por ambiente</button>
      </div>
      <div id="stats_tables" style="margin-top:8px;"></div>
      <details class="json-fold">
        <summary>Ver JSON cru (para auditoria / cópia)</summary>
        <pre class="out" id="stats_out" style="max-height:280px;">Aguardando execução…</pre>
      </details>
    </section>

    <!-- TAB: REGRESSION -->
    <section data-pane="reg" style="display:none;">
      <h2>ML supervisionado — Regressão</h2>
      <p class="field">Prediz <code>rsrp_dbm</code> a partir de variáveis ambientais,
         espaciais e de mobilidade. Modelos: regressão linear, RandomForest,
         XGBoost e SVR (com scaler). A validação agrupada é a estimativa
         científica principal; a aleatória é exibida apenas como contraste
         otimista. Sem <code>fillna(0)</code>.</p>
      <label class="field">Alvo</label>
      <select id="regTarget">
        <option value="rsrp_dbm">rsrp_dbm</option>
        <option value="sinr_db">sinr_db</option>
        <option value="ping_avg_ms">ping_avg_ms</option>
        <option value="test_dl_max_kbps">test_dl_max_kbps</option>
      </select>
      <label class="field">Features (opcional, vírgulas; vazio = padrão)</label>
      <input type="text" id="regFeatures" placeholder="ex: distance_to_serving_m,environment_class,sector_code">
      <label class="field">Validação</label>
      <select id="regGroup">
        <option value="campaign_id">Agrupada por campanha (principal)</option>
        <option value="sector_code_effective">Agrupada por setor</option>
        <option value="date">Agrupada por data</option>
        <option value="run_id">Agrupada por arquivo/run</option>
        <option value="random">Aleatória por linha (otimista)</option>
      </select>
      <button class="btn primary" id="btnReg">Treinar &amp; validar</button>
      <div id="reg_charts" style="margin-top:8px;"></div>
      <details class="json-fold">
        <summary>Ver JSON cru (para auditoria / cópia)</summary>
        <pre class="out" id="reg_out">Aguardando execução…</pre>
      </details>
    </section>

    <!-- TAB: CLASSIFICATION -->
    <section data-pane="cls" style="display:none;">
      <h2>ML supervisionado — Classificação</h2>
      <p class="field">Prediz as seis classes derivadas de RSRP — Excelente, Bom,
         Satisfatório, Ruim, Péssimo e Nulo — a partir
         de variáveis ambientais e espaciais — <i>sem</i> RSRP/RSRQ/SINR como features
         para evitar circularidade. Modelos: RandomForest, XGBoost, SVC (RBF + scaler).
         Validação agrupada por campanha, setor, data ou arquivo; a opção aleatória
         permanece apenas para contraste. Saída: matriz de confusão média,
         precisão/recall/F1 por classe.</p>
      <label class="field">Alvo</label>
      <select id="clsTarget">
        <option value="signal_rating">signal_rating (seis classes de RSRP)</option>
        <option value="environment_class">environment_class (edificado/aberto/arborizado)</option>
        <option value="network_tech">network_tech (5G/4G)</option>
      </select>
      <label class="field">Features (opcional, vírgulas; vazio = padrão)</label>
      <input type="text" id="clsFeatures" placeholder="ex: distance_to_serving_m,sector_code">
      <label class="field">Validação</label>
      <select id="clsGroup">
        <option value="campaign_id">Agrupada por campanha (principal)</option>
        <option value="sector_code_effective">Agrupada por setor</option>
        <option value="date">Agrupada por data</option>
        <option value="run_id">Agrupada por arquivo/run</option>
        <option value="random">Aleatória por linha (otimista)</option>
      </select>
      <button class="btn primary" id="btnCls">Treinar &amp; validar</button>
      <div id="cls_summary"></div>
      <details class="json-fold">
        <summary>Ver JSON cru (para auditoria / cópia)</summary>
        <pre class="out" id="cls_out">Aguardando execução…</pre>
      </details>
    </section>

    <!-- TAB: UNSUPERVISED -->
    <section data-pane="unsup" style="display:none;">
      <h2>ML não supervisionado — PCA, k-means e DBSCAN</h2>
      <p class="field">Caracteriza perfis empíricos dentro das rotas medidas.
         Variáveis numéricas são padronizadas; colunas com ≥50% de ausências
         são descartadas e as demais usam casos completos, sem imputação.
         O clima padrão usa apenas anotações de campo ou Open-Meteo Archive;
         valores legados do endpoint <code>current</code> são ignorados.
         Os clusters não representam generalização espacial fora das rotas.</p>
      <label class="field">Features numéricas (opcional, vírgulas; vazio = padrão)</label>
      <input type="text" id="unsupFeatures" placeholder="ex: rsrp_dbm,sinr_db,altitude_m,tree_count_eff">
      <button class="btn primary" id="btnUnsup">Executar PCA + clustering</button>
      <div id="unsup_summary" style="margin-top:8px;"></div>
      <details class="json-fold">
        <summary>Ver JSON cru (para auditoria / cópia)</summary>
        <pre class="out" id="unsup_out">Aguardando execução…</pre>
      </details>
    </section>

    <!-- TAB: AUDIT -->
    <section data-pane="audit" style="display:none;">
      <h2>Auditoria de sessões (runs)</h2>
      <button class="btn" id="btnRuns">Listar runs</button>
      <div class="scroll" id="runs_list"></div>
      <h2 style="margin-top:14px;">Exportar dataset</h2>
      <p class="field">Modo <b>científico</b> = só as colunas do Quadro 2 do plano.
         Modo <b>completo</b> = todas as colunas internas (run_id, status flags,
         timestamps de inserção, etc — útil para debugging).</p>
      <div class="row">
        <button class="btn primary" id="btnExportSci">Científico (.xlsx)</button>
        <button class="btn"          id="btnExportFull">Completo (.xlsx)</button>
      </div>
    </section>

    <!-- TAB: CALIBRATION -->
    <section data-pane="calib" style="display:none;">
      <h2>Calibração local → WGS84</h2>
      <p class="field">Transformação afim ajustada por mínimos quadrados a partir
         de pontos de controle medidos em campo. RMS reportado em metros sobre o
         elipsoide WGS84. Sem calibração ativa, todo ponto é classificado como
         <code>sector_code=null</code>.</p>
      <button class="btn" id="btnCalibStatus">Estado da calibração</button>
      <pre class="out" id="calib_status" style="max-height:160px;">Aguardando consulta…</pre>

      <label class="field">Pontos de controle (CSV ou JSON)</label>
      <p class="field" style="margin-top:0;">
        <b>CSV:</b> uma linha por ponto, colunas
        <code>name,x_local,y_local,lat,lon</code>.<br>
        <b>JSON:</b> array de objetos com as mesmas chaves.
      </p>
      <textarea id="calib_input" rows="6" style="width:100%;background:#0A1020;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-size:.72rem;font-family:'JetBrains Mono',monospace;" placeholder="name,x_local,y_local,lat,lon
CP1,32.5,25.0,-23.21000,-45.87800
CP2,265.0,350.0,-23.20850,-45.87650
CP3,330.0,105.0,-23.20950,-45.87600
CP4,675.0,210.0,-23.20880,-45.87420"></textarea>
      <label class="field">Notas (opcional)</label>
      <input type="text" id="calib_notes" placeholder="ex: Survey RTK 2026-05-04">
      <label class="field">RMS máximo aceitável (m, opcional)</label>
      <input type="number" id="calib_max_rms" step="0.1" placeholder="ex: 5.0">
      <button class="btn primary" id="btnCalibFit">Ajustar e salvar calibração</button>
      <button class="btn" id="btnReclassify">Reclassificar medições existentes</button>
      <pre class="out" id="calib_out" style="max-height:300px;">Aguardando ajuste…</pre>
    </section>

  </div>

  <!-- RIGHT PANEL — MAP -->
  <div class="panel" style="padding:6px;">
    <div class="map-shell">
      <div id="map"></div>
      <div class="controls">
        <span class="k">Camada:</span>
        <select id="layerSel">
          <option value="points">Pontos (RSRP)</option>
          <option value="rsrp">Heatmap RSRP</option>
          <option value="sinr">Heatmap SINR</option>
          <option value="latency">Heatmap latência</option>
          <option value="dl_throughput">Heatmap throughput DL</option>
          <option value="ul_throughput">Heatmap throughput UL</option>
          <option value="choro_rsrp">Choropleth setores · RSRP</option>
          <option value="choro_sinr">Choropleth setores · SINR</option>
          <option value="choro_latency">Choropleth setores · latência</option>
          <option value="choro_dl">Choropleth setores · DL kbps</option>
        </select>
        <span class="k">Tech:</span>
        <select id="techSel">
          <option value="">todos</option>
          <option value="5G">5G</option>
          <option value="4G">4G</option>
        </select>
        <span class="k">Amb:</span>
        <select id="envSel">
          <option value="">todos</option>
          <option value="edificado">edificado</option>
          <option value="aberto">aberto</option>
          <option value="arborizado">arborizado</option>
        </select>
        <label style="display:inline-flex;gap:4px;align-items:center;cursor:pointer;" title="Polígonos dos setores estão escondidos por padrão (calibração desalinhada).">
          <input type="checkbox" id="chkShowSectors"> setores
        </label>
        <button class="btn" id="btnRefreshMap" style="width:auto;padding:5px 10px;">Recarregar</button>
      </div>
      <div class="legend" id="mapLegend">
        <div><i style="background:#22C55E"></i>RSRP &gt; -85 (Excelente)</div>
        <div><i style="background:#84CC16"></i>-95 &lt; RSRP ≤ -85 (Bom)</div>
        <div><i style="background:#F59E0B"></i>-105 &lt; RSRP ≤ -95 (Satisfatório)</div>
        <div><i style="background:#EF4444"></i>-115 &lt; RSRP ≤ -105 (Ruim)</div>
        <div><i style="background:#B91C1C"></i>-125 &lt; RSRP ≤ -115 (Péssimo)</div>
        <div><i style="background:#6B7280"></i>RSRP ≤ -125 (Nulo)</div>
      </div>
    </div>
  </div>
</div>

<script>
const $  = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

// ----- Tabs -----------------------------------------------------------------
$$('.tab').forEach(t => t.addEventListener('click', () => {
  $$('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  $$('section[data-pane]').forEach(s => s.style.display='none');
  $(`section[data-pane=${t.dataset.tab}]`).style.display='block';
}));

// ----- Map ------------------------------------------------------------------
const map = L.map('map',{zoomControl:true}).setView([-22.8009, -45.1903], 17);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{
  attribution:'© OpenStreetMap & CartoDB',
  maxZoom:20
}).addTo(map);
let pointsLayer = L.layerGroup().addTo(map);
let heatLayer   = null;
let sectorLayer = null;

const ENV_COLOR = {
  edificado:  '#60A5FA',
  aberto:     '#FACC15',
  arborizado: '#22C55E',
  null:       '#94A3B8',
};

async function refreshSectors(){
  if (sectorLayer){ map.removeLayer(sectorLayer); sectorLayer = null; }
  // Polígonos escondidos por padrão (a calibração afim tem desalinhamento
  // conhecido com a geografia real do campus — poluem o mapa). Use o
  // checkbox "Mostrar polígonos dos setores" no canto do mapa para exibir.
  const show = (document.getElementById('chkShowSectors')||{}).checked;
  if (!show) return;
  try {
    const r = await fetch('/api/sectors');
    const fc = await r.json();
    if (!fc.properties || !fc.properties.calibrated) return;
    sectorLayer = L.geoJSON(fc, {
      style: f => ({
        color: ENV_COLOR[f.properties.environment_class ?? 'null'] || '#94A3B8',
        weight: 1.5, fillOpacity: 0.10,
      }),
      onEachFeature: (f, layer) => {
        const p = f.properties;
        layer.bindTooltip(`${p.sector_code} · ${p.sector_name}`, {sticky:true});
      },
    }).addTo(map);
  } catch(e){ /* uncalibrated → silent */ }
}

function colourForRSRP(v){
  if (v == null) return '#94A3B8';
  if (v > -85)  return '#22C55E';
  if (v > -95)  return '#84CC16';
  if (v > -105) return '#F59E0B';
  if (v > -115) return '#EF4444';
  if (v > -125) return '#B91C1C';
  return '#6B7280';
}

// --- Layer config ----------------------------------------------------------
// 'higher_is_better' tells the heatmap whether to invert weights so that
// "worse" values are always rendered hottest. Choropleth uses the same flag
// to decide the direction of the colour ramp.
const HEAT_LAYERS = {
  rsrp:           {key:'rsrp_dbm',         label:'RSRP (dBm)',        better:'higher'},
  sinr:           {key:'sinr_db',          label:'SINR (dB)',         better:'higher'},
  latency:        {key:'ping_avg_ms',      label:'Ping médio (ms)',   better:'lower'},
  dl_throughput:  {key:'test_dl_max_kbps', label:'Throughput DL (kbps)', better:'higher'},
  ul_throughput:  {key:'test_ul_max_kbps', label:'Throughput UL (kbps)', better:'higher'},
};
const CHORO_LAYERS = {
  choro_rsrp:    {metric:'rsrp_dbm',         label:'RSRP médio (dBm)',     better:'higher'},
  choro_sinr:    {metric:'sinr_db',          label:'SINR médio (dB)',      better:'higher'},
  choro_latency: {metric:'ping_avg_ms',      label:'Ping médio (ms)',      better:'lower'},
  choro_dl:      {metric:'test_dl_max_kbps', label:'DL kbps médio',        better:'higher'},
};

let choroLayer = null;

function rampColour(t, better){
  // t in [0..1]; "good" end = green, "bad" end = red, mid = amber.
  // If better === 'lower', invert t so smaller value is greener.
  const v = better === 'lower' ? 1 - t : t;
  if (v >= 0.66) return '#22C55E';
  if (v >= 0.33) return '#F59E0B';
  return '#E11D48';
}

function setLegend(html){ $('#mapLegend').innerHTML = html; }

function legendHeat(label, better){
  const a = better === 'lower' ? 'menor' : 'maior';
  const b = better === 'lower' ? 'maior' : 'menor';
  setLegend(
    `<b>${label}</b><br>`+
    `<i style="background:#22C55E"></i>${a} (melhor)<br>`+
    `<i style="background:#F59E0B"></i>intermediário<br>`+
    `<i style="background:#E11D48"></i>${b} (pior)`
  );
}

function legendChoro(label, scale, better){
  if (scale.min == null){
    setLegend(`<b>${label}</b><br>Sem setores com n≥3.`); return;
  }
  const fmt = v => (v==null?'—':v.toFixed(1));
  setLegend(
    `<b>${label} por setor</b><br>`+
    `min: <span class="num">${fmt(scale.min)}</span> · `+
    `p10: <span class="num">${fmt(scale.p10)}</span><br>`+
    `p90: <span class="num">${fmt(scale.p90)}</span> · `+
    `max: <span class="num">${fmt(scale.max)}</span><br>`+
    `<i style="background:#22C55E"></i>melhor &nbsp;`+
    `<i style="background:#F59E0B"></i>médio &nbsp;`+
    `<i style="background:#E11D48"></i>pior<br>`+
    `<span style="color:var(--muted);">cinza = sem dados (n&lt;3)</span>`
  );
}

function clearChoro(){
  if (choroLayer){ map.removeLayer(choroLayer); choroLayer = null; }
}

async function refreshMap(){
  const tech  = $('#techSel').value;
  const env   = $('#envSel').value;
  const layer = $('#layerSel').value;

  pointsLayer.clearLayers();
  if (heatLayer){ map.removeLayer(heatLayer); heatLayer = null; }

  // -------- Choropleth path: needs sectors + aggregates ---------------------
  if (CHORO_LAYERS[layer]){
    clearChoro();
    if (sectorLayer) map.removeLayer(sectorLayer);          // avoid double-fill
    const cfg = CHORO_LAYERS[layer];
    const [fcRes, agRes] = await Promise.all([
      fetch('/api/sectors'),
      fetch(`/api/sectors/aggregates?metric=${encodeURIComponent(cfg.metric)}`),
    ]);
    const fc = await fcRes.json();
    const ag = await agRes.json();
    if (!fc.properties || !fc.properties.calibrated){
      setLegend(`<b>${cfg.label}</b><br>Sem calibração ativa — abra a aba CALIBRAÇÃO.`);
      return;
    }
    const byCode = {};
    (ag.sectors || []).forEach(s => { byCode[s.sector_code] = s; });
    const scale = ag.scale || {min:null,max:null};
    const span = (scale.max != null && scale.min != null) ? (scale.max - scale.min) : 0;

    choroLayer = L.geoJSON(fc, {
      style: f => {
        const code = f.properties.sector_code;
        const s = byCode[code];
        if (!s || s.value == null){
          return {color:'#475569', weight:1, fillOpacity:0.10, fillColor:'#475569',
                  dashArray:'3 3'};
        }
        const t = span > 0 ? (s.value - scale.min) / span : 0.5;
        const col = rampColour(t, cfg.better);
        return {color:col, weight:1.5, fillColor:col, fillOpacity:0.55};
      },
      onEachFeature: (f, lyr) => {
        const code = f.properties.sector_code;
        const s = byCode[code] || {};
        lyr.bindTooltip(
          `<b>${code}</b> · ${f.properties.sector_name ?? ''}<br>`+
          `${cfg.label}: ${s.value == null ? '—' : s.value.toFixed(2)}<br>`+
          `n válidos: ${s.n_valid ?? 0} / ${s.n_rows ?? 0}<br>`+
          `ambiente: ${f.properties.environment_class ?? '—'}`,
          {sticky:true}
        );
      },
    }).addTo(map);
    legendChoro(cfg.label, scale, cfg.better);
    return;
  }

  // -------- Point/heatmap path: needs measurements --------------------------
  clearChoro();
  if (sectorLayer && !map.hasLayer(sectorLayer)) sectorLayer.addTo(map);
  const r = await fetch('/api/points?limit=10000');
  const data = await r.json();
  let filtered = data;
  if (tech) filtered = filtered.filter(d => d.network_tech === tech);
  if (env)  filtered = filtered.filter(
    d => (d.environment_class_effective ?? d.environment_class) === env
  );

  if (layer === 'points'){
    filtered.forEach(d => {
      const c = L.circleMarker([d.latitude, d.longitude],{
        radius:5, color:colourForRSRP(d.rsrp_dbm), fillOpacity:0.85, weight:1
      }).bindPopup(
        `<div style="font-size:.75rem;line-height:1.4;">`+
        `<b style="color:#0EA5A4;">SINAL</b><br>`+
        `RSRP: <b>${d.rsrp_dbm ?? '-'}</b> dBm · ${d.signal_rating ?? '-'}<br>`+
        `SINR: ${d.sinr_db ?? '-'} dB · CQI: ${d.cqi ?? '-'}<br>`+
        `Tech: ${d.network_tech ?? '-'} · ${d.band ?? '-'} · `+
          `${d.frequency_hz ? (d.frequency_hz/1e6).toFixed(0)+' MHz' : '-'}<br>`+
        `<b style="color:#0EA5A4;">QoS</b><br>`+
        `Ping: ${d.ping_avg_ms ?? '-'} ms · jitter ${d.ping_stdev_ms ?? '-'} ms<br>`+
        `DL: ${d.test_dl_max_kbps != null ? (d.test_dl_max_kbps/1000).toFixed(2)+' Mbps' : '-'}`+
        ` · UL: ${d.test_ul_max_kbps != null ? (d.test_ul_max_kbps/1000).toFixed(2)+' Mbps' : '-'}<br>`+
        `<b style="color:#0EA5A4;">CONTEXTO</b><br>`+
        `Setor: ${(d.sector_code_effective ?? d.sector_code_manual ?? d.sector_code) ?? '-'}<br>`+
        `Ambiente: ${(d.environment_class_effective ?? d.environment_class) ?? '-'} · ${d.indoor_outdoor ?? '-'}${d.surface_type ? ' · '+d.surface_type : ''}<br>`+
        `Altitude: ${d.altitude_m != null ? d.altitude_m.toFixed(0)+' m' : '-'}`+
          `${d.period_of_day ? ' · '+d.period_of_day : ''}<br>`+
        `Dist. à referência espacial: ${d.distance_to_site_est_m != null ? d.distance_to_site_est_m.toFixed(0)+' m' : '-'}<br>`+
        `<b style="color:#0EA5A4;">CLIMA</b><br>`+
        `Temp: ${d.temperature_c_eff ?? '-'} °C · Umid: ${d.humidity_eff ?? '-'} %<br>`+
        `Nuvens: ${d.cloud_cover_pct_eff != null ? d.cloud_cover_pct_eff+' %' : '-'}`+
          `${d.weather_source_eff ? ' · fonte '+d.weather_source_eff : ''}`+
          `${d.precipitation_status != null ? ' · chuva '+d.precipitation_status : ''}<br>`+
        `<b style="color:#0EA5A4;">EDIFÍCIOS / ÁRVORES</b><br>`+
        `Prédios: ${(d.building_count_eff ?? d.building_count) ?? '-'} (h̄ ${(d.avg_building_height_eff_m ?? d.avg_building_height) != null ? (d.avg_building_height_eff_m ?? d.avg_building_height).toFixed(1)+' m' : '-'}, `+
          `d̄ ${(d.distance_to_building_m_eff ?? d.distance_to_building_m) != null ? (d.distance_to_building_m_eff ?? d.distance_to_building_m).toFixed(1)+' m' : '-'})<br>`+
        `Árvores: ${(d.tree_count_eff ?? d.tree_count) ?? '-'} (h̄ ${(d.avg_tree_height_eff_m ?? d.avg_tree_height_m) != null ? (d.avg_tree_height_eff_m ?? d.avg_tree_height_m).toFixed(1)+' m' : '-'}, `+
          `d̄ ${(d.distance_to_tree_m_eff ?? d.distance_to_tree_m) != null ? (d.distance_to_tree_m_eff ?? d.distance_to_tree_m).toFixed(1)+' m' : '-'})<br>`+
        `NDVI: ${d.tree_density_ndvi != null ? d.tree_density_ndvi.toFixed(3) : (d.vegetation_density_manual != null ? 'densidade '+d.vegetation_density_manual+'/3 (manual)' : '-')}<br>`+
        `<span style="color:#94A3B8;font-size:.65rem;">GPS acc ${d.gps_accuracy_m ?? '-'} m (${d.gps_quality ?? '-'}) · ${d.campaign_id ?? 'sem campanha'}</span>`+
        `</div>`
      );
      pointsLayer.addLayer(c);
    });
    setLegend(
      `<b>Pontos · cor = RSRP</b><br>`+
      `<i style="background:#22C55E"></i>RSRP &gt; -85 (Excelente)<br>`+
      `<i style="background:#84CC16"></i>-95 &lt; RSRP ≤ -85 (Bom)<br>`+
      `<i style="background:#F59E0B"></i>-105 &lt; RSRP ≤ -95 (Satisfatório)<br>`+
      `<i style="background:#EF4444"></i>-115 &lt; RSRP ≤ -105 (Ruim)<br>`+
      `<i style="background:#B91C1C"></i>-125 &lt; RSRP ≤ -115 (Péssimo)<br>`+
      `<i style="background:#6B7280"></i>RSRP ≤ -125 (Nulo)<br>`+
      `<span style="color:var(--muted)">${filtered.length} pts</span>`
    );
    return;
  }

  const cfg = HEAT_LAYERS[layer];
  if (!cfg) return;
  const vals = filtered.map(d => d[cfg.key]).filter(v => v != null);
  if (!vals.length){
    setLegend(`<b>${cfg.label}</b><br>Sem amostras com a métrica.`); return;
  }
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const heat = filtered.filter(d => d[cfg.key] != null).map(d => {
    let w = (d[cfg.key] - lo) / (hi - lo + 1e-9);
    if (cfg.better === 'higher') w = 1 - w;   // pior fica mais quente
    return [d.latitude, d.longitude, w];
  });
  heatLayer = L.heatLayer(heat, {radius:24, blur:18, maxZoom:19}).addTo(map);
  legendHeat(cfg.label, cfg.better);
  if (filtered.length){
    map.fitBounds(filtered.map(d=>[d.latitude,d.longitude]),{padding:[20,20]});
  }
}
$('#btnRefreshMap').addEventListener('click', refreshMap);
$('#layerSel').addEventListener('change', refreshMap);
$('#chkShowSectors').addEventListener('change', refreshSectors);
$('#techSel').addEventListener('change', refreshMap);
$('#envSel').addEventListener('change', refreshMap);

// ----- Upload --------------------------------------------------------------
$('#btnUpload').addEventListener('click', async () => {
  const files = Array.from($('#logIn').files || []);
  if (!files.length){ alert('Selecione um ou mais arquivos .txt/.csv'); return; }
  $('#ingest_out').style.display='none';
  const params = new URLSearchParams();
  params.set('enrich', $('#enrich').checked ? 'true' : 'false');
  const cid = $('#campaignIn').value.trim();
  if (cid) params.set('campaign_id', cid);
  const io  = $('#ioSel').value;
  if (io) params.set('indoor_outdoor', io);
  const ms = $('#manualSectorSel').value;
  if (ms) params.set('manual_sector', ms);
  const surf = $('#surfaceSel').value;
  if (surf) params.set('surface_type', surf);
  const bh = $('#bldHeightIn').value.trim();
  if (bh) params.set('avg_building_height_m', bh);
  const th = $('#treeHeightIn').value.trim();
  if (th) params.set('avg_tree_height_m', th);
  const bc = $('#bldCountIn').value.trim();   if (bc) params.set('building_count', bc);
  const bd = $('#bldDistIn').value.trim();    if (bd) params.set('distance_to_building_m', bd);
  const tc = $('#treeCountIn').value.trim();  if (tc) params.set('tree_count', tc);
  const td = $('#treeDistIn').value.trim();   if (td) params.set('distance_to_tree_m', td);
  const tp = $('#tempIn').value.trim();       if (tp) params.set('temperature_c', tp);
  const hm = $('#humIn').value.trim();        if (hm) params.set('humidity', hm);
  const cv = $('#cloudIn').value.trim();      if (cv) params.set('cloud_cover_pct', cv);
  const vd = $('#vegDensSel').value;          if (vd !== '') params.set('vegetation_density', vd);
  const pr = $('#precipSel').value;           if (pr !== '') params.set('precipitation_status', pr);
  const ob = $('#obsSel').value;              if (ob !== '') params.set('visual_obstruction_grade', ob);
  if ($('#forceUp').checked) params.set('force', 'true');

  const results = [];
  for (let i=0; i<files.length; i++){
    const f = files[i];
    $('#ingest_dump').textContent =
      `Enviando ${i+1}/${files.length}: ${f.name} (${(f.size/1024).toFixed(1)} KB)…`;
    const fd = new FormData(); fd.append('file', f);
    const res = await fetch('/api/upload?' + params.toString(),
                            {method:'POST', body:fd});
    results.push({file: f.name, status: res.status, body: await res.json()});
  }
  $('#ingest_dump').textContent = JSON.stringify(results, null, 2);
  const warns = results.flatMap(r => r.body.warnings || []);
  const dupes = results.filter(r => r.body.status === 'already_ingested');
  if (dupes.length || warns.length){
    $('#ingest_out').style.display='block';
    $('#ingest_out').textContent =
      (dupes.length ? `${dupes.length} arquivo(s) já estavam ingeridos. Marque "Forçar reingestão" para sobrescrever.\n` : '') +
      warns.join('\n');
  }
  refreshMap();
  refreshSummary();
});

$('#btnEnrichExisting').addEventListener('click', async () => {
  $('#ingest_dump').textContent = 'Enriquecendo medições já ingeridas (apenas as sem dados ambientais)…';
  const r = await fetch('/api/enrich?only_missing=true', {method:'POST'});
  const d = await r.json();
  $('#ingest_dump').textContent = JSON.stringify(d, null, 2);
  refreshSummary();
});

$('#btnCampaigns').addEventListener('click', async () => {
  const r = await fetch('/api/campaigns');
  const d = await r.json();
  if (!d.campaigns || !d.campaigns.length){
    $('#ingest_dump').textContent = 'Nenhuma campanha registrada ainda.';
    return;
  }
  let html = `<table><thead><tr><th>campaign_id</th><th>runs</th><th>n</th><th>setores</th><th>primeiro</th><th>último</th></tr></thead><tbody>`;
  d.campaigns.forEach(c => {
    html += `<tr><td>${c.campaign_id ?? '<i style="color:#94A3B8">—</i>'}</td>`+
            `<td class="num">${c.n_runs}</td>`+
            `<td class="num">${c.n_measurements}</td>`+
            `<td class="num">${c.n_distinct_sectors}</td>`+
            `<td>${c.time_first?.replace('T',' ').slice(0,19) ?? '—'}</td>`+
            `<td>${c.time_last?.replace('T',' ').slice(0,19) ?? '—'}</td></tr>`;
  });
  html += '</tbody></table>';
  $('#ingest_dump').innerHTML = html;
});

$('#btnTempCov').addEventListener('click', async () => {
  const r = await fetch('/api/sectors/temporal_coverage');
  const d = await r.json();
  if (d.error){ $('#ingest_dump').textContent = d.error; return; }
  let html = `<p class="field" style="color:var(--warn);">${d.policy}</p>`;
  html += `<table><thead><tr><th>setor</th><th>n</th><th>campanhas</th><th>horas distintas</th><th>horas</th></tr></thead><tbody>`;
  d.sectors.forEach(s => {
    const flag = s.n_distinct_hours <= 1 ? ' style="color:#F59E0B;font-weight:700"' : '';
    html += `<tr${flag}><td>${s.sector_code}</td>`+
            `<td class="num">${s.n_measurements}</td>`+
            `<td class="num">${s.n_campaigns}</td>`+
            `<td class="num">${s.n_distinct_hours}</td>`+
            `<td class="num">${s.hours.join(', ') || '—'}</td></tr>`;
  });
  html += '</tbody></table>';
  $('#ingest_dump').innerHTML = html;
});

// ----- Stats ---------------------------------------------------------------
function fmt(v, digits=3){
  if (v === null || v === undefined) return '—';
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return '—';
  // p-values get scientific notation when very small
  if (Math.abs(v) > 0 && Math.abs(v) < 1e-3) return v.toExponential(2);
  return v.toFixed(digits);
}

function fmtPVal(p){
  if (p === null || p === undefined || !isFinite(p)) return '—';
  const cls = p < 0.001 ? 'ok' : (p < 0.05 ? '' : 'bad');
  const s = p < 1e-3 ? p.toExponential(2) : p.toFixed(4);
  return `<span class="${cls}">${s}</span>`;
}

function renderAnovaTables(d){
  const host = $('#stats_tables');
  host.innerHTML = '';
  const a = d.anova;
  if (!a || a.error){
    host.innerHTML = a && a.error
      ? `<div class="warn">${a.error}</div>` : '';
    return;
  }

  // ---- Header line ------------------------------------------------------
  let html = `<h3 style="margin:8px 0 4px;font-size:.78rem;color:var(--accent);">`+
             `ANOVA — ${a.response} ~ ${a.factor}`+
             `${a.n_total ? ` <span class="badge">N=${a.n_total} · ${a.n_groups} grupos</span>` : ''}`+
             `</h3>`;

  // ---- Robust path: three tests + diagnostics + effect size ------------
  if (a.tests){
    const t = a.tests, es = a.effect_size || {};
    html += `<table><thead><tr><th>teste</th><th>estatística</th><th>p</th><th>df</th></tr></thead><tbody>`+
      `<tr><td>F clássico</td><td class="num">${fmt(t.anova_classic.F)}</td><td class="num">${fmtPVal(t.anova_classic.p)}</td><td class="num">${t.anova_classic.df1},${t.anova_classic.df2}</td></tr>`+
      `<tr><td>F de Welch</td><td class="num">${fmt(t.anova_welch.F)}</td><td class="num">${fmtPVal(t.anova_welch.p)}</td><td class="num">${fmt(t.anova_welch.df1,1)},${fmt(t.anova_welch.df2,1)}</td></tr>`+
      `<tr><td>Kruskal-Wallis</td><td class="num">H=${fmt(t.kruskal_wallis.H)}</td><td class="num">${fmtPVal(t.kruskal_wallis.p)}</td><td class="num">${t.kruskal_wallis.df}</td></tr>`+
      `</tbody></table>`;
    html += `<p class="field"><b>Tamanho de efeito:</b> η²=<span class="num">${fmt(es.eta_squared)}</span> · ω²=<span class="num">${fmt(es.omega_squared)}</span></p>`;
    if (a.diagnostics){
      const dg = a.diagnostics;
      html += `<p class="field"><b>Levene:</b> W=${fmt(dg.levene_W)} p=${fmtPVal(dg.levene_p)}`+
              ` · ${dg.heteroscedastic_at_alpha ? '<span class="bad">variâncias DESIGUAIS</span>' : '<span class="ok">variâncias homogêneas</span>'}</p>`;
      const nonNorm = (dg.shapiro_per_group||[]).filter(s=>s.non_normal_at_alpha).map(s=>s.level);
      if (nonNorm.length){
        html += `<p class="field"><b>Shapiro-Wilk:</b> grupos não-normais (p&lt;.05): <span class="bad">${nonNorm.join(', ')}</span></p>`;
      }
    }
    if (a.recommendation){
      html += `<div class="warn" style="white-space:pre-wrap;">Recomendação: ${a.recommendation}</div>`;
    }
  } else {
    // Classic ANOVA: F, p, eta²
    html += `<p class="field"><b>F=</b><span class="num">${fmt(a.F)}</span> · `+
            `<b>p=</b>${fmtPVal(a.p)} · η²=<span class="num">${fmt(a.eta_squared)}</span></p>`;
  }

  // ---- Group summary -------------------------------------------------
  if (a.groups && a.groups.length){
    html += `<h4 style="margin:10px 0 4px;font-size:.74rem;color:var(--muted);text-transform:uppercase;">Grupos</h4>`;
    html += `<table><thead><tr><th>nível</th><th>n</th><th>média</th><th>std</th><th>mediana</th><th>p25</th><th>p75</th>`+
            (a.groups[0].ci95_low !== undefined ? `<th>CI95</th>` : '')+
            `</tr></thead><tbody>`;
    a.groups.forEach(g => {
      let ci = '';
      if (g.ci95_low !== undefined){
        ci = `<td class="num">[${fmt(g.ci95_low,2)}, ${fmt(g.ci95_high,2)}]</td>`;
      }
      html += `<tr><td>${g.level}</td><td class="num">${g.n}</td>`+
              `<td class="num">${fmt(g.mean,2)}</td><td class="num">${fmt(g.std,2)}</td>`+
              `<td class="num">${fmt(g.median,2)}</td>`+
              `<td class="num">${fmt(g.p25,2)}</td><td class="num">${fmt(g.p75,2)}</td>${ci}</tr>`;
    });
    html += `</tbody></table>`;
  }

  // ---- Tukey HSD (classic mode) -----------------------------------------
  if (a.tukey_hsd && a.tukey_hsd.length){
    html += `<h4 style="margin:10px 0 4px;font-size:.74rem;color:var(--muted);text-transform:uppercase;">Tukey HSD</h4>`;
    html += `<table><thead><tr><th>g1</th><th>g2</th><th>Δ média</th><th>p ajust.</th><th>IC95</th><th>rejeita H₀</th></tr></thead><tbody>`;
    a.tukey_hsd.forEach(t => {
      const cls = t.reject ? 'ok' : '';
      html += `<tr class="${cls}"><td>${t.group1}</td><td>${t.group2}</td>`+
              `<td class="num">${fmt(t.meandiff,2)}</td>`+
              `<td class="num">${fmtPVal(t.p_adj)}</td>`+
              `<td class="num">[${fmt(t.lower,2)}, ${fmt(t.upper,2)}]</td>`+
              `<td>${t.reject ? '<span class="ok">SIM</span>' : 'não'}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  // ---- Hedges' g (robust mode) ------------------------------------------
  const hg = a.pairwise_hedges_g;
  if (hg && hg.pairs && hg.pairs.length){
    html += `<h4 style="margin:10px 0 4px;font-size:.74rem;color:var(--muted);text-transform:uppercase;">Hedges' g (com correção de viés)</h4>`;
    html += `<table><thead><tr><th>g1</th><th>g2</th><th>n1/n2</th><th>g</th><th>IC95</th><th>magnitude</th></tr></thead><tbody>`;
    hg.pairs.forEach(p => {
      const sig = (p.ci95_low !== null && p.ci95_high !== null
                   && (p.ci95_low > 0 || p.ci95_high < 0));
      const cls = sig ? 'ok' : '';
      html += `<tr class="${cls}"><td>${p.group1}</td><td>${p.group2}</td>`+
              `<td class="num">${p.n1}/${p.n2}</td>`+
              `<td class="num">${fmt(p.g)}</td>`+
              `<td class="num">[${fmt(p.ci95_low)}, ${fmt(p.ci95_high)}]</td>`+
              `<td>${p.magnitude}</td></tr>`;
    });
    html += `</tbody></table>`;
  }

  // ---- MDE (robust mode) ------------------------------------------------
  const mde = a.minimum_detectable_effect;
  if (mde && mde.rows && mde.rows.length){
    html += `<h4 style="margin:10px 0 4px;font-size:.74rem;color:var(--muted);text-transform:uppercase;">MDE (poder = ${mde.power}, α=${mde.alpha})</h4>`;
    html += `<table><thead><tr><th>nível</th><th>n</th><th>n contraparte</th><th>menor d detectável</th><th>piso</th></tr></thead><tbody>`;
    mde.rows.forEach(r => {
      const big = r.min_detectable_d != null && r.min_detectable_d > 0.8;
      const cls = big ? 'bad' : '';
      html += `<tr class="${cls}"><td>${r.level}</td><td class="num">${r.n}</td>`+
              `<td class="num">${r.n_companion}</td>`+
              `<td class="num">${fmt(r.min_detectable_d)}</td>`+
              `<td>${r.magnitude_floor}</td></tr>`;
    });
    html += `</tbody></table>`;
    html += `<p class="field" style="color:var(--muted);">${mde.interpretation}</p>`;
  }

  // ---- Pearson FDR matrix ----------------------------------------------
  const pf = d.pearson_fdr;
  if (pf && pf.cols && pf.cols.length){
    html += `<h4 style="margin:10px 0 4px;font-size:.74rem;color:var(--muted);text-transform:uppercase;">Correlações de Pearson (FDR-BH)</h4>`;
    html += `<table><thead><tr><th></th>`+
            pf.cols.map(c=>`<th title="${c}">${c.length>10?c.slice(0,9)+'…':c}</th>`).join('')+
            `</tr></thead><tbody>`;
    for (let i=0;i<pf.cols.length;i++){
      html += `<tr><th>${pf.cols[i]}</th>`;
      for (let j=0;j<pf.cols.length;j++){
        const r = pf.r[i][j];
        const rej = pf.rejected_fdr[i][j];
        const cell = r === null ? '—' : r;
        const cls = rej ? (Number(r) > 0 ? 'ok' : 'bad') : '';
        html += `<td class="num ${cls}">${cell}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
    html += `<p class="field" style="color:var(--muted);">Verde/vermelho = sobrevive ao FDR-BH (α=${pf.alpha}).</p>`;
  }

  host.innerHTML = html;
}

$('#btnStats').addEventListener('click', async () => {
  const f = $('#anovaFactor').value;
  const r = $('#anovaResponse').value;
  const robust = $('#statsRobust').checked ? '&robust=true' : '';
  $('#stats_out').textContent = 'Calculando…';
  $('#stats_tables').innerHTML = '';
  const res = await fetch(`/api/statistics?factor=${encodeURIComponent(f)}&response=${encodeURIComponent(r)}${robust}`);
  const d = await res.json();
  $('#stats_out').textContent = JSON.stringify(d, null, 2);
  renderAnovaTables(d);
});

function renderSectorEnvTable(host, data, isEnv){
  host.innerHTML = '';
  const list = isEnv ? data.environments : data.sectors;
  if (!list || !list.length){
    host.textContent = data.error || 'Sem dados.';
    return;
  }
  const note = document.createElement('div');
  note.style.fontSize = '.7rem';
  note.style.color = 'var(--muted)';
  note.style.marginBottom = '6px';
  note.textContent = 'Coluna usada: ' + (data.source_column || '(padrão)');
  host.appendChild(note);
  const tbl = document.createElement('table');
  tbl.className = 'dash-table';
  const head = isEnv
    ? '<tr><th>Ambiente</th><th>n</th><th>setores</th><th>RSRP médio</th><th>SINR médio</th><th>Ping médio</th></tr>'
    : '<tr><th>Setor</th><th>Nome</th><th>Ambiente</th><th>n</th><th>RSRP médio</th><th>SINR médio</th><th>Ping médio</th></tr>';
  tbl.innerHTML = '<thead>' + head + '</thead>';
  const tb = document.createElement('tbody');
  for (const it of list){
    const m = it.metrics || {};
    const rsrp = (m.rsrp_dbm && m.rsrp_dbm.mean != null) ? m.rsrp_dbm.mean.toFixed(1) : '—';
    const sinr = (m.sinr_db  && m.sinr_db.mean  != null) ? m.sinr_db.mean.toFixed(1)  : '—';
    const ping = (m.ping_avg_ms && m.ping_avg_ms.mean != null) ? m.ping_avg_ms.mean.toFixed(1) : '—';
    const tr = document.createElement('tr');
    if (isEnv){
      tr.innerHTML = `<td><b>${it.environment_class}</b></td><td>${it.n_rows}</td><td>${it.n_distinct_sectors ?? '—'}</td><td>${rsrp}</td><td>${sinr}</td><td>${ping}</td>`;
    } else {
      tr.innerHTML = `<td><b>${it.sector_code}</b></td><td>${it.sector_name ?? '—'}</td><td>${it.environment_class ?? '—'}</td><td>${it.n_rows}</td><td>${rsrp}</td><td>${sinr}</td><td>${ping}</td>`;
    }
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  host.appendChild(tbl);
}

$('#btnBySector').addEventListener('click', async () => {
  $('#stats_out').textContent = 'Agregando por setor…';
  $('#stats_tables').innerHTML = '';
  const r = await fetch('/api/statistics/by_sector');
  const d = await r.json();
  $('#stats_out').textContent = '';
  renderSectorEnvTable($('#stats_tables'), d, false);
});

$('#btnByEnv').addEventListener('click', async () => {
  $('#stats_out').textContent = 'Agregando por ambiente…';
  $('#stats_tables').innerHTML = '';
  const r = await fetch('/api/statistics/by_environment');
  const d = await r.json();
  $('#stats_out').textContent = '';
  renderSectorEnvTable($('#stats_tables'), d, true);
});

// ============================================================================
// ----- DASHBOARD (aba "Painel de resultados") -------------------------------
// ============================================================================
function kpiCard(label, value, sub, klass){
  return `<div class="kpi-card ${klass||''}">
            <div class="lbl">${label}</div>
            <div class="val">${value}</div>
            ${sub ? `<div class="sub">${sub}</div>` : ''}
          </div>`;
}

function fmtNum(v, dec){ return (v == null || isNaN(v)) ? '—' : Number(v).toFixed(dec ?? 1); }

function renderDashKPIs(host, k){
  if (!k){ host.innerHTML = ''; return; }
  const rsrp = k.rsrp || {};
  const sinr = k.sinr || {};
  const ping = k.ping || {};
  const dl   = k.dl_kbps || {};
  const ul   = k.ul_kbps || {};
  const techShare = (k.n_5g + k.n_4g) > 0
                    ? Math.round(100*k.n_5g / (k.n_5g + k.n_4g))
                    : null;
  const rsrpClass = rsrp.mean == null ? '' : (rsrp.mean > -85 ? 'good' : rsrp.mean > -105 ? 'warn' : 'bad');
  host.innerHTML = [
    kpiCard('Medições analíticas', k.n_measurements ?? '—',
            k.n_measurements_raw != null
              ? `${k.n_measurements_raw} linhas brutas · ${k.n_dropped_exact_duplicates ?? 0} duplicatas removidas`
              : ''),
    kpiCard('Campanhas',    k.n_campaigns ?? '—'),
    kpiCard('Setores',      k.n_sectors ?? '—'),
    kpiCard('5G / 4G',      `${k.n_5g}/${k.n_4g}`, techShare != null ? `${techShare}% em 5G` : ''),
    kpiCard('RSRP médio',   fmtNum(rsrp.mean) + ' dBm', `min ${fmtNum(rsrp.min)} · máx ${fmtNum(rsrp.max)}`, rsrpClass),
    kpiCard('SINR médio',   fmtNum(sinr.mean) + ' dB',  `mediana ${fmtNum(sinr.median)}`),
    kpiCard('Ping médio',   ping.mean != null ? fmtNum(ping.mean) + ' ms' : '—', ping.n ? `${ping.n} ciclos` : 'sem Data Sequence'),
    kpiCard('Download méd', dl.mean != null ? fmtNum(dl.mean/1024, 2) + ' Mbps' : '—', dl.max ? `pico ${fmtNum(dl.max/1024, 1)} Mbps` : ''),
    kpiCard('Upload méd',   ul.mean != null ? fmtNum(ul.mean/1024, 2) + ' Mbps' : '—', ul.max ? `pico ${fmtNum(ul.max/1024, 1)} Mbps` : ''),
  ].join('');
}

function renderDashCharts(host, d){
  host.innerHTML = '';
  // ----- Histograma RSRP ---------------------------------------------------
  if (d.rsrp_histogram && d.rsrp_histogram.bins){
    const wrap = document.createElement('div');
    wrap.className = 'dash-section';
    wrap.innerHTML = '<h3>Distribuição de RSRP</h3><div class="dash-plot" id="plot_hist"></div>';
    host.appendChild(wrap);
    Plotly.newPlot('plot_hist', [{
      x: d.rsrp_histogram.bins,
      y: d.rsrp_histogram.counts,
      type: 'bar',
      marker: {color: '#0EA5A4'},
    }], {
      paper_bgcolor:'#0A1020', plot_bgcolor:'#0A1020',
      font:{color:'#E8ECF3', family:'Inter, sans-serif', size:10},
      margin:{l:40, r:20, t:10, b:50},
      xaxis:{title:'RSRP (dBm)', tickangle:-25},
      yaxis:{title:'medições', gridcolor:'#1F2A44'},
      height:220,
    }, {displayModeBar:false, responsive:true});
  }

  // ----- Pizza signal_rating ----------------------------------------------
  if (d.signal_rating && d.signal_rating.length){
    const wrap = document.createElement('div');
    wrap.className = 'dash-section';
    wrap.innerHTML = '<h3>Qualidade do sinal (categorias)</h3><div class="dash-plot" id="plot_rating"></div>';
    host.appendChild(wrap);
    const palette = {'Excelente':'#22C55E','Bom':'#84CC16','Satisfatório':'#F59E0B','Ruim':'#EF4444','Péssimo':'#B91C1C','Nulo':'#6B7280'};
    Plotly.newPlot('plot_rating', [{
      type:'pie', hole: 0.55,
      labels: d.signal_rating.map(x => x.label),
      values: d.signal_rating.map(x => x.n),
      marker:{colors: d.signal_rating.map(x => palette[x.label] || '#94A3B8')},
      textinfo:'label+percent',
    }], {
      paper_bgcolor:'#0A1020',
      font:{color:'#E8ECF3', family:'Inter, sans-serif', size:10},
      margin:{l:10, r:10, t:10, b:10},
      height:230, showlegend:false,
    }, {displayModeBar:false, responsive:true});
  }

  // ----- Comparação por categoria ------------------------------------------
  const cats = [
    {key:'by_tech',        title:'RSRP médio por tecnologia'},
    {key:'by_band',        title:'RSRP médio por banda'},
    {key:'by_environment', title:'RSRP médio por ambiente'},
    {key:'by_surface',     title:'RSRP médio por superfície'},
    {key:'by_period',      title:'RSRP médio por período do dia'},
    {key:'by_sector',      title:'RSRP médio por setor'},
  ];
  for (const c of cats){
    const arr = d[c.key];
    if (!arr || !arr.length) continue;
    const wrap = document.createElement('div');
    wrap.className = 'dash-section';
    const id = 'plot_' + c.key;
    wrap.innerHTML = `<h3>${c.title}</h3><div class="dash-plot" id="${id}"></div>`;
    host.appendChild(wrap);
    Plotly.newPlot(id, [{
      type:'bar',
      x: arr.map(a => a.group),
      y: arr.map(a => a.mean),
      error_y:{type:'data', array: arr.map(a => (a.p75 - a.p25)/2), visible:true, color:'#94A3B8'},
      text: arr.map(a => `n=${a.n}`),
      textposition:'outside',
      marker:{color:'#0EA5A4'},
    }], {
      paper_bgcolor:'#0A1020', plot_bgcolor:'#0A1020',
      font:{color:'#E8ECF3', family:'Inter, sans-serif', size:10},
      margin:{l:50, r:20, t:10, b:50},
      xaxis:{tickangle: arr.length > 4 ? -25 : 0},
      yaxis:{title:'RSRP médio (dBm)', gridcolor:'#1F2A44'},
      height:230,
    }, {displayModeBar:false, responsive:true});
  }

  // ----- Série temporal RSRP -----------------------------------------------
  if (d.timeline && d.timeline.length > 1){
    const wrap = document.createElement('div');
    wrap.className = 'dash-section';
    wrap.innerHTML = '<h3>Evolução do RSRP médio (por minuto)</h3><div class="dash-plot" id="plot_timeline"></div>';
    host.appendChild(wrap);
    Plotly.newPlot('plot_timeline', [{
      type:'scatter', mode:'lines+markers',
      x: d.timeline.map(p => p.t),
      y: d.timeline.map(p => p.rsrp_mean),
      line:{color:'#0EA5A4', width:2},
      marker:{size:5, color:'#0EA5A4'},
    }], {
      paper_bgcolor:'#0A1020', plot_bgcolor:'#0A1020',
      font:{color:'#E8ECF3', family:'Inter, sans-serif', size:10},
      margin:{l:50, r:20, t:10, b:50},
      xaxis:{title:'data/hora', tickangle:-25},
      yaxis:{title:'RSRP médio (dBm)', gridcolor:'#1F2A44'},
      height:230,
    }, {displayModeBar:false, responsive:true});
  }
}

function renderDashTables(host, d){
  host.innerHTML = '';
  if (!d.by_sector || !d.by_sector.length) return;
  const wrap = document.createElement('div');
  wrap.className = 'dash-section';
  wrap.innerHTML = '<h3>Tabela detalhada por setor</h3>';
  const tbl = document.createElement('table');
  tbl.className = 'dash-table';
  tbl.innerHTML = '<thead><tr><th>Setor</th><th>n</th><th>RSRP médio</th><th>p25</th><th>mediana</th><th>p75</th></tr></thead>';
  const tb = document.createElement('tbody');
  for (const it of d.by_sector){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><b>${it.group}</b></td><td>${it.n}</td>
                    <td>${fmtNum(it.mean)}</td><td>${fmtNum(it.p25)}</td>
                    <td>${fmtNum(it.median)}</td><td>${fmtNum(it.p75)}</td>`;
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  wrap.appendChild(tbl);
  host.appendChild(wrap);
}

async function refreshDashboard(){
  $('#dash_loading').style.display = 'block';
  $('#dash_kpis').innerHTML = '';
  $('#dash_charts').innerHTML = '';
  $('#dash_tables').innerHTML = '';
  const cid = $('#dashCampaign').value;
  const url = '/api/dashboard' + (cid ? '?campaign_id=' + encodeURIComponent(cid) : '');
  try {
    const r = await fetch(url);
    const d = await r.json();
    if (d.empty){
      $('#dash_kpis').innerHTML = '<div style="color:var(--muted);font-size:.8rem;">Banco vazio. Suba um log primeiro na aba INGESTÃO.</div>';
    } else {
      renderDashKPIs($('#dash_kpis'), d.kpis);
      // Ajuste descritivo relativo a uma referência espacial estimada/declarada.
      if (d.propagation){
        const p = d.propagation;
        let extra = kpiCard('Inclinação log-distância', p.path_loss_exponent.toFixed(2),
                            `Descritiva · R²=${p.r2.toFixed(2)} · ${p.n_points} pts · referência ${p.site_source ?? 'estimada'}`);
        if (p.o2i_db != null){
          extra += kpiCard('Diferença indoor–outdoor', p.o2i_db.toFixed(1) + ' dB',
                           'Diferença descritiva entre grupos; não é perda de penetração pareada', 'warn');
        }
        $('#dash_kpis').insertAdjacentHTML('beforeend', extra);
      }
      renderDashCharts($('#dash_charts'), d);
      renderDashTables($('#dash_tables'), d);
    }
  } catch(e){
    $('#dash_kpis').innerHTML = `<div style="color:var(--danger);">Erro: ${e.message}</div>`;
  } finally {
    $('#dash_loading').style.display = 'none';
  }
}

async function populateDashCampaigns(){
  try {
    const r = await fetch('/api/campaigns');
    const d = await r.json();
    const sel = $('#dashCampaign');
    while (sel.options.length > 1) sel.remove(1);
    const items = (d.campaigns || []).sort((a,b) => String(a.campaign_id||'').localeCompare(String(b.campaign_id||'')));
    for (const c of items){
      if (!c.campaign_id) continue;
      const opt = document.createElement('option');
      opt.value = c.campaign_id;
      opt.textContent = `${c.campaign_id} (n=${c.n_measurements ?? c.n ?? '?'})`;
      sel.appendChild(opt);
    }
  } catch(e){ console.warn('Falha ao popular campanhas:', e); }
}

$('#btnDashRefresh').addEventListener('click', refreshDashboard);
$('#dashCampaign').addEventListener('change', refreshDashboard);

// Atualiza dashboard ao clicar na aba
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    if (t.dataset.tab === 'dashboard'){
      populateDashCampaigns().then(refreshDashboard);
    }
  });
});

// ----- Regression ----------------------------------------------------------
function renderRegressionCharts(d){
  const host = $('#reg_charts');
  host.innerHTML = '';
  if (!d.models || !d.models.length) return;

  // ----- Bar chart: R² mean ± std per model (baseline as reference line) ---
  const r2_div = document.createElement('div');
  r2_div.id = 'plot_r2';
  r2_div.style.height = '260px';
  host.appendChild(r2_div);

  const baseline = d.models.find(m => m.is_baseline);
  const baseR2 = baseline ? baseline.summary.R2.mean : 0;
  const names = d.models.map(m => m.name);
  const means = d.models.map(m => m.summary.R2.mean);
  const stds  = d.models.map(m => m.summary.R2.std);
  const colors = d.models.map(m => m.is_baseline ? '#F59E0B' : '#0EA5A4');
  Plotly.newPlot(r2_div, [{
    x: names, y: means, type: 'bar', marker: {color: colors},
    error_y: {type:'data', array: stds, visible:true, color:'#94A3B8'},
    text: means.map(v => v.toFixed(3)),
    textposition: 'outside',
  }], {
    title: {text: `R² por modelo (${d.target}) — baseline = ${baseR2.toFixed(3)}`,
            font:{size:13, color:'#E8ECF3'}},
    margin:{t:35,b:50,l:50,r:10},
    paper_bgcolor:'#152034', plot_bgcolor:'#152034',
    font:{color:'#CBD5E1'}, yaxis:{zeroline:true},
    shapes:[{type:'line', x0:-0.5, x1:names.length-0.5, y0:baseR2, y1:baseR2,
             line:{color:'#F59E0B', dash:'dash', width:1.5}}],
  }, {displayModeBar:false, responsive:true});

  // ----- Residual scatter + histogram for the best non-baseline model ------
  const best = d.models.filter(m => !m.is_baseline)
                       .sort((a,b)=>b.summary.R2.mean - a.summary.R2.mean)[0];
  if (!best || !best.residual_diagnostics) return;
  const rd = best.residual_diagnostics;

  const sc_div = document.createElement('div');
  sc_div.id = 'plot_scatter';
  sc_div.style.height = '300px';
  host.appendChild(sc_div);
  const yMin = rd.stats.y_true_min, yMax = rd.stats.y_true_max;
  Plotly.newPlot(sc_div, [
    {x: rd.scatter.y_true, y: rd.scatter.y_pred, mode:'markers',
     type:'scatter', marker:{size:5, color:'#0EA5A4', opacity:0.7},
     name: 'OOF predict'},
    {x:[yMin,yMax], y:[yMin,yMax], mode:'lines', type:'scatter',
     line:{color:'#94A3B8', dash:'dash'}, name:'y=ŷ ideal',
     showlegend:false},
  ], {
    title:{text: `Predito vs real (out-of-fold) — ${best.name} ` +
                  `· MAE=${rd.stats.mae.toFixed(2)} RMSE=${rd.stats.rmse.toFixed(2)}`,
           font:{size:12, color:'#E8ECF3'}},
    xaxis:{title:'Real ('+d.target+')'}, yaxis:{title:'Predito'},
    margin:{t:35,b:45,l:55,r:10},
    paper_bgcolor:'#152034', plot_bgcolor:'#152034', font:{color:'#CBD5E1'},
  }, {displayModeBar:false, responsive:true});

  // Histogram of residuals
  const hist_div = document.createElement('div');
  hist_div.id = 'plot_hist';
  hist_div.style.height = '220px';
  host.appendChild(hist_div);
  const edges = rd.histogram.edges;
  const centres = edges.slice(0,-1).map((e,i)=>(e+edges[i+1])/2);
  const widths  = edges.slice(0,-1).map((e,i)=>edges[i+1]-e);
  Plotly.newPlot(hist_div, [{
    x: centres, y: rd.histogram.counts, width: widths,
    type:'bar', marker:{color:'#0EA5A4'},
  }], {
    title:{text: `Histograma de resíduos (μ=${rd.stats.residual_mean.toFixed(2)}, ` +
                  `σ=${rd.stats.residual_std.toFixed(2)})` +
                  (rd.shapiro && rd.shapiro.non_normal_at_0_05
                   ? ' · Shapiro-Wilk: NÃO-NORMAL (p<0.05)'
                   : ''),
           font:{size:12, color:'#E8ECF3'}},
    xaxis:{title:'resíduo (real − predito)'},
    yaxis:{title:'contagem'},
    margin:{t:35,b:45,l:55,r:10},
    paper_bgcolor:'#152034', plot_bgcolor:'#152034', font:{color:'#CBD5E1'},
    shapes:[{type:'line', x0:0, x1:0, y0:0, y1:Math.max(...rd.histogram.counts),
             line:{color:'#F59E0B', dash:'dash'}}],
  }, {displayModeBar:false, responsive:true});
}

$('#btnReg').addEventListener('click', async () => {
  const t = $('#regTarget').value;
  const f = $('#regFeatures').value.trim();
  const g = $('#regGroup').value;
  $('#reg_out').textContent = 'Treinando e validando modelos; o agrupamento preserva rotas inteiras…';
  $('#reg_charts').innerHTML = '';
  const q = `target=${encodeURIComponent(t)}&group_by=${encodeURIComponent(g)}` +
            (f ? `&features=${encodeURIComponent(f)}` : '');
  const res = await fetch(`/api/ml/regression?${q}`);
  const d = await res.json();
  $('#reg_out').textContent = JSON.stringify(d, null, 2);
  if (!d.error) renderRegressionCharts(d);
});

// ----- Classification ------------------------------------------------------
function renderConfusion(model){
  const labels = model.confusion_matrix_labels || [];
  const cm     = model.confusion_matrix_mean    || [];
  if (!labels.length) return '';
  const tag = model.is_baseline ? ' <span class="badge" style="background:#241B0A;color:#FCD34D;border-color:#F59E0B;">BASELINE</span>' : '';
  let html = `<h3 style="margin:8px 0 4px;font-size:.78rem;color:var(--accent);">Matriz de confusão (média entre folds) — ${model.name}${tag}</h3>`;
  html += `<table><thead><tr><th></th>${labels.map(l=>`<th>pred ${l}</th>`).join('')}</tr></thead><tbody>`;
  cm.forEach((row, i) => {
    html += `<tr><th>real ${labels[i]}</th>`;
    row.forEach(v => { html += `<td class="num">${v.toFixed(2)}</td>`; });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  if (model.classification_report){
    html += `<h3 style="margin:8px 0 4px;font-size:.78rem;color:var(--accent);">Métricas por classe</h3>`;
    html += `<table><thead><tr><th>classe</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr></thead><tbody>`;
    Object.entries(model.classification_report).forEach(([cls, m]) => {
      html += `<tr><td>${cls}</td><td class="num">${m.precision.toFixed(3)}</td><td class="num">${m.recall.toFixed(3)}</td><td class="num">${m.f1.toFixed(3)}</td><td class="num">${m.support_total_test}</td></tr>`;
    });
    html += `</tbody></table>`;
  }
  return html;
}

$('#btnCls').addEventListener('click', async () => {
  const t = $('#clsTarget').value;
  const f = $('#clsFeatures').value.trim();
  const g = $('#clsGroup').value;
  $('#cls_out').textContent = 'Treinando e validando classificadores com grupos inteiros…';
  $('#cls_summary').innerHTML = '';
  const q = `target=${encodeURIComponent(t)}&group_by=${encodeURIComponent(g)}` +
            (f ? `&features=${encodeURIComponent(f)}` : '');
  const res = await fetch(`/api/ml/classification?${q}`);
  const d = await res.json();
  $('#cls_out').textContent = JSON.stringify(d, null, 2);
  if (d.models){
    // Render baselines first (informative floor) then real models
    const ordered = [...d.models].sort((a,b)=>(a.is_baseline?-1:0)-(b.is_baseline?-1:0));
    $('#cls_summary').innerHTML = ordered.map(renderConfusion).join('<hr style="border:0;border-top:1px solid var(--line);margin:10px 0;">');
  }
});

// ----- PCA + clustering ---------------------------------------------------
function renderUnsupervised(d){
  const host = $('#unsup_summary');
  host.innerHTML = '';
  if (d.error){ host.innerHTML = `<div class="warn">${d.error}</div>`; return; }

  const audit = d.audit || {};
  const pca = d.pca || {};
  const km = d.kmeans || {};
  const db = d.dbscan || {};
  host.innerHTML = `
    <div class="kpi-grid">
      <div class="kpi"><div class="v">${audit.rows_complete_case ?? '-'}</div><div class="l">casos completos</div></div>
      <div class="kpi"><div class="v">${audit.features_used?.length ?? '-'}</div><div class="l">variáveis usadas</div></div>
      <div class="kpi"><div class="v">${km.selected_k ?? '-'}</div><div class="l">k-means · k escolhido</div></div>
      <div class="kpi"><div class="v">${db.selected?.n_clusters_excluding_noise ?? '-'}</div><div class="l">DBSCAN · clusters</div></div>
    </div>
    <div class="warn" style="margin-top:8px;">${d.scope_label}</div>`;

  const scree = document.createElement('div');
  scree.style.height = '230px'; host.appendChild(scree);
  const variance = pca.explained_variance_ratio || [];
  Plotly.newPlot(scree, [{
    x: variance.map((_,i)=>`PC${i+1}`),
    y: variance.map(v=>100*v), type:'bar', marker:{color:'#0EA5A4'},
  }], {
    title:{text:'Variância explicada por componente',font:{size:12,color:'#E8ECF3'}},
    yaxis:{title:'variância explicada (%)'}, margin:{t:35,b:45,l:55,r:10},
    paper_bgcolor:'#152034', plot_bgcolor:'#152034', font:{color:'#CBD5E1'},
  }, {displayModeBar:false, responsive:true});

  const points = d.points_sample || [];
  if (points.length){
    const scatter = document.createElement('div');
    scatter.style.height = '300px'; host.appendChild(scatter);
    const labels = [...new Set(points.map(p=>p.kmeans_cluster))].sort((a,b)=>a-b);
    const traces = labels.map(label => {
      const subset = points.filter(p=>p.kmeans_cluster===label);
      return {
        x:subset.map(p=>p.pc1), y:subset.map(p=>p.pc2), mode:'markers',
        type:'scattergl', name:`cluster ${label}`, marker:{size:5,opacity:.65},
        text:subset.map(p=>`${p.campaign_id ?? ''} · ${p.sector_code_effective ?? ''}`),
        hovertemplate:'PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>%{text}<extra></extra>',
      };
    });
    Plotly.newPlot(scatter, traces, {
      title:{text:'PCA · pontos coloridos pelo k-means',font:{size:12,color:'#E8ECF3'}},
      xaxis:{title:'PC1'}, yaxis:{title:'PC2'}, margin:{t:35,b:45,l:55,r:10},
      paper_bgcolor:'#152034', plot_bgcolor:'#152034', font:{color:'#CBD5E1'},
    }, {displayModeBar:false, responsive:true});
  }

  const loadings = (pca.loadings || []).slice(0,3);
  if (loadings.length){
    let html = '<h3>Maiores cargas do PCA</h3><table><thead><tr><th>componente</th><th>variância</th><th>cargas principais</th></tr></thead><tbody>';
    loadings.forEach(c => {
      const terms = c.top_loadings.map(x=>`${x.feature}: ${x.loading.toFixed(3)}`).join('<br>');
      html += `<tr><td>${c.component}</td><td class="num">${(100*c.explained_variance_ratio).toFixed(1)}%</td><td>${terms}</td></tr>`;
    });
    html += '</tbody></table>';
    const block = document.createElement('div'); block.innerHTML = html; host.appendChild(block);
  }

  const dbNote = document.createElement('div');
  dbNote.className = db.warning ? 'warn' : 'ok';
  dbNote.style.marginTop = '8px';
  dbNote.textContent = db.warning ||
    `DBSCAN: eps=${db.selected?.eps?.toFixed(3)}, ruído=${(100*(db.selected?.noise_fraction || 0)).toFixed(1)}%.`;
  host.appendChild(dbNote);
}

$('#btnUnsup').addEventListener('click', async () => {
  const f = $('#unsupFeatures').value.trim();
  $('#unsup_out').textContent = 'Padronizando variáveis e executando PCA, k-means e DBSCAN…';
  $('#unsup_summary').innerHTML = '';
  const q = f ? `?features=${encodeURIComponent(f)}` : '';
  const res = await fetch('/api/ml/unsupervised' + q);
  const d = await res.json();
  $('#unsup_out').textContent = JSON.stringify(d, null, 2);
  renderUnsupervised(d);
});

// ----- Audit ---------------------------------------------------------------
$('#btnRuns').addEventListener('click', async () => {
  const r = await fetch('/api/audit/runs');
  const d = await r.json();
  if (!d.length){ $('#runs_list').textContent = 'Sem runs ainda.'; return; }
  let html = `<table><thead><tr><th>id</th><th>arq</th><th>variant</th><th>raw</th><th>ess</th><th>gps</th><th>valid</th><th>quando</th></tr></thead><tbody>`;
  d.forEach(r => {
    html += `<tr><td><a href="javascript:void(0)" onclick="showRun(${r.id})">${r.id}</a></td><td>${r.filename}</td><td><span class="badge">${r.variant}</span></td><td class="num">${r.rows_raw}</td><td class="num">${r.rows_dropped_essential}</td><td class="num">${r.rows_dropped_gps}</td><td class="num ok">${r.rows_valid}</td><td>${r.started_at?.replace('T',' ').slice(0,19) ?? '-'}</td></tr>`;
  });
  html += '</tbody></table>';
  $('#runs_list').innerHTML = html;
});

async function showRun(id){
  const r = await fetch('/api/audit/run/'+id);
  const d = await r.json();
  $('#runs_list').innerHTML += `<pre class="out">${JSON.stringify(d,null,2)}</pre>`;
}
window.showRun = showRun;

// ----- Export --------------------------------------------------------------
function exportXlsx(mode){
  window.location.href = '/api/export?mode=' + mode;
}
['btnExportSci','hdrExportSci'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', () => exportXlsx('scientific'));
});
['btnExportFull','hdrExportFull'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', () => exportXlsx('full'));
});

// ----- Calibration ---------------------------------------------------------
function parseControlPoints(raw){
  const txt = (raw || '').trim();
  if (!txt) return [];
  // JSON form
  if (txt.startsWith('[') || txt.startsWith('{')){
    const j = JSON.parse(txt);
    return Array.isArray(j) ? j : [j];
  }
  // CSV form
  const lines = txt.split(/\r?\n/).map(s=>s.trim()).filter(s=>s && !s.startsWith('#'));
  if (!lines.length) return [];
  let header = ['name','x_local','y_local','lat','lon'];
  let start  = 0;
  const first = lines[0].split(',').map(s=>s.trim());
  if (first.includes('name') && first.includes('lat') && first.includes('lon')){
    header = first; start = 1;
  }
  const out = [];
  for (let i=start; i<lines.length; i++){
    const cols = lines[i].split(',').map(s=>s.trim());
    if (cols.length < 5) continue;
    const obj = {};
    header.forEach((h,j)=>{ obj[h] = cols[j]; });
    obj.x_local = parseFloat(obj.x_local);
    obj.y_local = parseFloat(obj.y_local);
    obj.lat     = parseFloat(obj.lat);
    obj.lon     = parseFloat(obj.lon);
    out.push(obj);
  }
  return out;
}

$('#btnCalibStatus').addEventListener('click', async () => {
  const r = await fetch('/api/sectors/calibration');
  const d = await r.json();
  $('#calib_status').textContent = JSON.stringify(d, null, 2);
});

$('#btnCalibFit').addEventListener('click', async () => {
  let cps;
  try { cps = parseControlPoints($('#calib_input').value); }
  catch(e){ $('#calib_out').textContent = 'Erro de parse: '+e.message; return; }
  if (cps.length < 3){ $('#calib_out').textContent = 'Forneça ≥3 pontos.'; return; }
  const body = {
    control_points: cps,
    notes: $('#calib_notes').value || '',
  };
  const mr = parseFloat($('#calib_max_rms').value);
  if (!isNaN(mr)) body.max_rms_m = mr;

  $('#calib_out').textContent = 'Ajustando…';
  const res = await fetch('/api/sectors/calibration', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body),
  });
  const d = await res.json();
  $('#calib_out').textContent = JSON.stringify(d, null, 2);
  if (res.ok){ refreshSectors(); refreshMap(); refreshSummary(); }
});

$('#btnReclassify').addEventListener('click', async () => {
  $('#calib_out').textContent = 'Reclassificando…';
  const res = await fetch('/api/sectors/reclassify', {method:'POST'});
  const d = await res.json();
  $('#calib_out').textContent = JSON.stringify(d, null, 2);
  if (res.ok){ refreshMap(); refreshSummary(); }
});

// ----- Expand/collapse left panel -----------------------------------------
$('#btnExpand').addEventListener('click', () => {
  const layout = document.querySelector('.layout');
  const expanded = layout.classList.toggle('expanded');
  $('#btnExpand').textContent = expanded ? '⇲' : '⛶';
  $('#btnExpand').title = expanded
    ? 'Voltar para o layout dividido (mapa visível)'
    : 'Expandir painel (oculta o mapa)';
  // Plotly resize so charts use the new width.
  setTimeout(() => {
    document.querySelectorAll('[id^="plot_"]').forEach(el => {
      try { Plotly.Plots.resize(el); } catch(e){}
    });
  }, 50);
});

// ----- Header status badge -------------------------------------------------
async function refreshSummary(){
  try {
    const r = await fetch('/api/summary');
    const s = await r.json();
    // Calibration banner
    const banner = $('#cal_banner');
    if (s.calibration === 'synthetic'){
      banner.style.display = 'block';
      banner.innerHTML = '⚠️ <b>Calibração sintética/de teste detectada</b> — '+
        'classificações por setor não são confiáveis. Vá na aba CALIBRAÇÃO, '+
        'cole pontos de controle MEDIDOS NO CAMPO e clique "Ajustar e salvar".';
    } else {
      banner.style.display = 'none';
    }
    const calColor = s.calibration === 'calibrated' ? 'var(--good)' :
                     s.calibration === 'synthetic'  ? 'var(--warn)' :
                                                       'var(--muted)';
    const calLabel = s.calibration === 'calibrated' ? 'calibrada' :
                     s.calibration === 'synthetic'  ? 'SINTÉTICA · refazer!' :
                                                       'sem calibração';
    $('#hdr_summary').innerHTML =
      `<span class="num">${s.n_measurements_analytical ?? s.n_measurements}</span> analíticas `+
      `(${s.n_measurements_raw ?? s.n_measurements} brutas) · `+
      `<span class="num">${s.n_campaigns}</span> campanhas · `+
      `<span class="num">${s.n_runs}</span> runs<br>`+
      `<span class="num">${s.n_classified}</span> com setor · `+
      `<span class="num">${s.n_weather_valid ?? s.n_enriched}</span> com clima válido · `+
      `<span style="color:${calColor};font-weight:700;">${calLabel}</span>`;
  } catch(e){
    $('#hdr_summary').textContent = 'erro ao consultar /api/summary';
  }
}

// ----- Marcador da referência espacial experimental ------------------------
let siteMarker = null;
async function refreshSiteMarker(){
  try {
    const r = await fetch('/api/site/estimate');
    const d = await r.json();
    if (!d || !d.available) return;
    if (siteMarker){ map.removeLayer(siteMarker); siteMarker = null; }
    const icon = L.divIcon({
      html: '<div style="font-size:26px;line-height:26px;' +
            'filter:drop-shadow(0 1px 3px rgba(0,0,0,.8));">📡</div>',
      className: '', iconSize: [26, 26], iconAnchor: [13, 13],
    });
    const isManual = d.source === 'manual';
    const titulo = isManual ? '📡 REFERÊNCIA — posição declarada'
                            : '📡 REFERÊNCIA — posição estimada';
    const rodape = isManual
      ? (d.notes ? 'Nota: ' + d.notes : 'Posição declarada pelo pesquisador; verificar associação LTE↔NR.')
      : 'Posição inferida dos próprios dados de RSRP; não identifica nem comprova a coordenada oficial da torre.';
    const modelo = (d.intercept_dbm != null)
      ? `<b style="color:#0EA5A4;">Ajuste log-distância descritivo (${d.band ?? '-'})</b><br>` +
        `RSRP = ${d.intercept_dbm} − 10·<b>${d.path_loss_exponent}</b>·log₁₀(d)<br>` +
        `R² = ${d.r2} · ${d.n_points_fit} medições · d: ${d.distance_min_m}–${d.distance_max_m} m<br>`
      : '';
    siteMarker = L.marker([d.lat, d.lon], {icon, zIndexOffset: 1000})
      .bindPopup(
        `<div style="font-size:.75rem;line-height:1.45;">` +
        `<b style="color:${isManual ? '#22C55E' : '#E11D48'};">${titulo}</b><br>` +
        `lat ${d.lat.toFixed(6)} · lon ${d.lon.toFixed(6)}<br>` +
        modelo +
        `<span style="color:#94A3B8;font-size:.65rem;">${rodape}</span>` +
        `</div>`
      ).addTo(map);
  } catch(e){ /* sem estimativa → sem marcador */ }
}

// ----- Popula o dropdown de setor declarado -------------------------------
async function populateManualSectors(){
  try {
    const r = await fetch('/api/sectors');
    const gj = await r.json();
    const sel = $('#manualSectorSel');
    // Limpa opcoes salvo a primeira ("nao informar")
    while (sel.options.length > 1) sel.remove(1);
    const feats = (gj.features || [])
      .map(f => f.properties || {})
      .filter(p => p.sector_code)
      .sort((a, b) => String(a.sector_code).localeCompare(String(b.sector_code)));
    // Categorias especiais (ambientes que cruzam varios setores)
    for (const sp of [
      {code:'VIA', name:'Via / Rua (carros passam)', env:'via'},
      {code:'EST', name:'Estacionamento',            env:'aberto'},
    ]){
      const opt = document.createElement('option');
      opt.value = sp.code;
      opt.textContent = `${sp.code} — ${sp.name} (${sp.env})`;
      sel.appendChild(opt);
    }
    for (const p of feats){
      const opt = document.createElement('option');
      opt.value = p.sector_code;
      opt.textContent = `${p.sector_code} — ${p.sector_name || ''}`
                       + (p.environment_class ? ` (${p.environment_class})` : '');
      sel.appendChild(opt);
    }
  } catch(e){
    console.warn('Falha ao popular setores manuais:', e);
  }
}

// Initial map load
refreshSectors();
refreshMap();
refreshSummary();
populateManualSectors();
refreshSiteMarker();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn, webbrowser
    init_db()
    url = f"http://{settings.host}:{settings.port}"
    print(f"FEG-UNESP RF Research Platform — {url}")
    try:
        webbrowser.open(url)
    except Exception:                                   # noqa: BLE001
        pass
    uvicorn.run("app.main:app", host=settings.host, port=settings.port,
                reload=False, log_config=None)
