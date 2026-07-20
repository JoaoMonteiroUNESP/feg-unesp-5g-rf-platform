"""
Classificador por setor com zona de influencia (buffer em metros).

Motivacao
---------
O classificador estrito (`classifier.py`) so devolve um setor quando o ponto
GPS cai estritamente dentro do retangulo do prédio. Mas na propagação 5G,
um ponto a poucos metros da fachada sofre praticamente o mesmo bloqueio/
reflexao que um ponto colado nela. Excluir esses pontos da analise por setor
é cientificamente equivocado.

Este modulo introduz uma classificação em DOIS niveis:

* `sector_code_strict`: como o classificador antigo — somente dentro do
  poligono original (alta confianca, "ponto comprovadamente dentro").
* `sector_code_buffer`: setor mais proximo, se a distancia ate a borda do
  poligono e <= buffer da classe ambiental daquele setor.

E uma variavel continua:

* `sector_distance_m`: distancia em metros do ponto ate a borda do poligono
  buffer (negativa quando dentro, positiva fora). Pode entrar como feature
  em modelos.

Buffers por classe ambiental (justificados em §15.x do plano):
  edificado   -> 15 m (≈ altura tipica de predio de 2-3 andares)
  aberto      ->  5 m (sem obstrucao significativa)
  arborizado  -> 10 m (copas geram zona de influencia intermediaria)
  outros      -> 10 m (default conservador)

Toda a geometria roda em metros LOCAIS (sistema do desenho), nao em graus —
isso so e possivel porque temos a calibracao afim. Use `classify_wgs84` se
voce so tem (lat, lon).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from shapely.geometry import Point, box
from shapely.strtree import STRtree

from app.sectors.calibration import Calibration, transform_wgs84_to_local
from app.sectors.legend import LEGEND, code_for


# Buffers em metros locais. Ajustar aqui e re-rodar reclassify_all
# para que toda a base seja atualizada.
BUFFER_M_POR_CLASSE: dict[str | None, float] = {
    "edificado":   15.0,
    "aberto":       5.0,
    "arborizado":  10.0,
    "a_confirmar": 10.0,
    None:          10.0,
}


@dataclass(frozen=True)
class BufferedHit:
    sector_code_strict:       str | None
    sector_name_strict:       str | None
    environment_class_strict: str | None
    sector_code_buffer:       str | None
    sector_name_buffer:       str | None
    environment_class_buffer: str | None
    sector_distance_m:        float | None  # negativa se dentro


_NULL = BufferedHit(None, None, None, None, None, None, None)


class BufferedSectorClassifier:
    """
    Trabalha com retangulos em metros locais. Recebe pontos em metros locais
    (x, y) ou em WGS84 via helper.
    """

    def __init__(self, sectors_local: Iterable[dict]):
        self._polys = []
        # cada meta: (sector_id, sector_code, sector_name, env_class, buffer_m)
        self._meta: list[tuple[int, str, str, str | None, float]] = []
        for s in sectors_local:
            poly = box(s["xmin"], s["ymin"], s["xmax"], s["ymax"])
            meta = LEGEND.get(s["id"])
            env_class = meta.environment_class if meta else None
            buf = BUFFER_M_POR_CLASSE.get(env_class, BUFFER_M_POR_CLASSE[None])
            name = meta.name if meta else f"Setor {s['id']}"
            self._polys.append(poly)
            self._meta.append((s["id"], code_for(s["id"]), name, env_class, buf))
        self._tree: STRtree | None = STRtree(self._polys) if self._polys else None

    @property
    def n_sectors(self) -> int:
        return len(self._polys)

    # -----------------------------------------------------------------
    def classify_local(self, x_local: float | None, y_local: float | None
                       ) -> BufferedHit:
        """Classifica um ponto ja em metros locais."""
        if self._tree is None or x_local is None or y_local is None:
            return _NULL
        try:
            pt = Point(float(x_local), float(y_local))
        except (TypeError, ValueError):
            return _NULL

        # ---- Nivel 1: estrito ----
        strict_code = strict_name = strict_env = None
        for i in self._tree.query(pt):
            poly = self._polys[int(i)]
            if poly.covers(pt):
                _, code, name, env, _ = self._meta[int(i)]
                strict_code, strict_name, strict_env = code, name, env
                break

        # ---- Nivel 2: buffer com nearest ----
        # Distancia ate cada poligono em metros (negativa se dentro).
        best_dist = float("inf")
        best_idx: int | None = None
        for idx in range(len(self._polys)):
            poly = self._polys[idx]
            _, _, _, _, buf = self._meta[idx]
            if poly.covers(pt):
                # Dentro: distancia "ate a borda interna" e negativa.
                # Aproximamos pela distancia ate o boundary do poligono.
                dist_m = -poly.exterior.distance(pt)
            else:
                dist_m = poly.distance(pt)
            # Considera apenas se cabe no buffer deste setor
            if dist_m <= buf and dist_m < best_dist:
                best_dist = dist_m
                best_idx = idx

        if best_idx is None:
            return BufferedHit(
                sector_code_strict=strict_code,
                sector_name_strict=strict_name,
                environment_class_strict=strict_env,
                sector_code_buffer=None,
                sector_name_buffer=None,
                environment_class_buffer=None,
                sector_distance_m=None,
            )

        _, b_code, b_name, b_env, _ = self._meta[best_idx]
        return BufferedHit(
            sector_code_strict=strict_code,
            sector_name_strict=strict_name,
            environment_class_strict=strict_env,
            sector_code_buffer=b_code,
            sector_name_buffer=b_name,
            environment_class_buffer=b_env,
            sector_distance_m=float(best_dist),
        )

    # -----------------------------------------------------------------
    def classify_wgs84(self, lat: float | None, lon: float | None,
                       calibration: Calibration) -> BufferedHit:
        """Helper: converte (lat, lon) para metros locais e classifica."""
        if lat is None or lon is None:
            return _NULL
        try:
            x, y = transform_wgs84_to_local(calibration, float(lat), float(lon))
        except Exception:                                       # noqa: BLE001
            return _NULL
        return self.classify_local(x, y)

    # -----------------------------------------------------------------
    def classify_batch_wgs84(self,
                             points: Iterable[tuple[float | None, float | None]],
                             calibration: Calibration) -> list[BufferedHit]:
        return [self.classify_wgs84(la, lo, calibration) for la, lo in points]
