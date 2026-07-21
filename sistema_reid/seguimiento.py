"""Seguimiento temporal simple por distancia entre centros de cajas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .caracteristicas import Caja


@dataclass
class Pista:
    """Representa una persona seguida entre frames."""

    id_pista: int
    caja: Caja
    identidad: str = "desconocido"
    frames_perdidos: int = 0
    historial: list[Caja] = field(default_factory=list)


def calcular_centro(caja: Caja) -> Tuple[float, float]:
    """Calcula el centro de una caja delimitadora."""
    x1, y1, x2, y2 = caja
    # Comentario clave: el centro permite asociar detecciones entre frames cercanos.
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distancia_centros(caja_a: Caja, caja_b: Caja) -> float:
    """Calcula la distancia euclidiana entre dos centros de cajas."""
    ax, ay = calcular_centro(caja_a)
    bx, by = calcular_centro(caja_b)
    # Comentario clave: menor distancia indica mayor probabilidad de ser la misma persona.
    return float(np.hypot(ax - bx, ay - by))


class RastreadorSimple:
    """Asocia detecciones entre frames usando el centro de la caja."""

    def __init__(self, distancia_maxima: float = 80.0, max_frames_perdidos: int = 10) -> None:
        self.distancia_maxima = distancia_maxima
        self.max_frames_perdidos = max_frames_perdidos
        self.siguiente_id = 1
        self.pistas: Dict[int, Pista] = {}

    def actualizar(self, cajas: list[Caja], identidades: list[str] | None = None) -> Dict[int, Pista]:
        """Actualiza las pistas con las cajas detectadas en el frame actual."""
        identidades = identidades or ["desconocido"] * len(cajas)
        asignadas = set()

        for caja, identidad in zip(cajas, identidades):
            mejor_id = None
            mejor_distancia = float("inf")

            for id_pista, pista in self.pistas.items():
                distancia = distancia_centros(caja, pista.caja)
                if distancia < mejor_distancia and distancia <= self.distancia_maxima:
                    mejor_id = id_pista
                    mejor_distancia = distancia

            if mejor_id is None:
                # Comentario clave: si no existe una pista cercana, se crea una nueva identidad temporal.
                mejor_id = self.siguiente_id
                self.siguiente_id += 1
                self.pistas[mejor_id] = Pista(id_pista=mejor_id, caja=caja)

            pista = self.pistas[mejor_id]
            pista.caja = caja
            pista.identidad = identidad
            pista.frames_perdidos = 0
            pista.historial.append(caja)
            asignadas.add(mejor_id)

        for id_pista in list(self.pistas.keys()):
            if id_pista not in asignadas:
                self.pistas[id_pista].frames_perdidos += 1
                if self.pistas[id_pista].frames_perdidos > self.max_frames_perdidos:
                    del self.pistas[id_pista]

        return self.pistas
