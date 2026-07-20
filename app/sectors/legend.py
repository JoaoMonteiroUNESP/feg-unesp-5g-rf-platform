"""
Mapping between sector index (Excel rows) and PDF legend.

Sectors 1..17 correspond directly to the labels in the FEG-UNESP map PDF.
Sectors 18..21 exist in the Excel `Dados quadrantes` sheet but have no
explicit label in the legend; they are real building footprints in the
drawing that need ground-truth confirmation. They are flagged
`to_verify=True` and intentionally have `environment_class=None` until
verified in field — this preserves auditability.

Environment classes
-------------------
* edificado   — primarily man-made structures and their immediate envelope
* aberto      — open ground, sports fields, parking
* arborizado  — vegetation-dominated areas (NDVI-validated when GEE available)
* a_confirmar — placeholder for the four un-labelled sectors
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorMeta:
    name: str
    environment_class: str | None
    to_verify: bool = False


LEGEND: dict[int, SectorMeta] = {
    1:  SectorMeta("Bloco 1",          "edificado"),
    2:  SectorMeta("Bloco 2",          "edificado"),
    3:  SectorMeta("Bloco 3",          "edificado"),
    4:  SectorMeta("Bloco 4",          "edificado"),
    5:  SectorMeta("FEGÃO",            "edificado"),
    6:  SectorMeta("Campo de Futebol", "aberto"),
    7:  SectorMeta("Creche",           "edificado"),
    8:  SectorMeta("Biblioteca",       "edificado"),
    9:  SectorMeta("Lab. Materiais",   "edificado"),
    10: SectorMeta("Dpto. Civil",      "edificado"),
    11: SectorMeta("IPBEN",            "edificado"),
    12: SectorMeta("CTIG",             "edificado"),
    13: SectorMeta("Bloco 5",          "edificado"),
    14: SectorMeta("DFI",              "edificado"),
    15: SectorMeta("DMA",              "edificado"),
    16: SectorMeta("INOVEE",           "edificado"),
    17: SectorMeta("Moradia",          "edificado"),
    # Setores confirmados em 2026-06-08 por inspeção do João Guilherme:
    # S18, S20 e S21 são áreas externas tipo calçada/pátio, sem árvores
    # de porte. S19 é a única com vegetação mais densa entre as quatro.
    # S21 corresponde a um campo de futebol secundário (predominância de
    # grama, sem prédios ou árvores ao redor).
    18: SectorMeta("Área externa S (entre Civil e IPBEN)",   "aberto"),
    19: SectorMeta("Área arborizada central (perto do DFI)", "arborizado"),
    20: SectorMeta("Área externa N (perto da Biblioteca)",   "aberto"),
    21: SectorMeta("Campo de futebol secundário (SW)",       "aberto"),
}


# Categorias especiais que NAO sao poligonos do mapa — sao AMBIENTES que
# cruzam varios setores (ruas, estacionamentos). O aluno declara pelo campo
# "setor declarado" no upload. Codigo em texto (nao S01..S21).
# Adicionado 2026-07-07: vias/ruas sao um ambiente de propagacao proprio
# (corredor aberto, asfalto, arborizado, carros como refletores moveis).
SPECIAL_SECTORS: dict[str, SectorMeta] = {
    "VIA": SectorMeta("Via / Rua (carros passam)", "via"),
    "EST": SectorMeta("Estacionamento",            "aberto"),
}


def code_for(idx: int) -> str:
    """Stable, sortable code for a sector ('S01' .. 'S21')."""
    return f"S{idx:02d}"
