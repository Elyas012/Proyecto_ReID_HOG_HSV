"""Pruebas básicas de extracción de características."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sistema_reid.caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, recortar_torso, rostro_es_util


def test_extraer_hog_rostro():
    imagen = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    vector = extraer_hog_rostro(imagen)
    assert vector.ndim == 1
    assert vector.size > 0


def test_extraer_histograma_hsv():
    imagen = np.random.randint(0, 255, (220, 120, 3), dtype=np.uint8)
    torso = recortar_torso(imagen)
    vector = extraer_histograma_hsv(torso)
    assert vector.ndim == 1
    assert vector.size == 16 * 16 * 8


def test_rostro_es_util_devuelve_booleano():
    imagen = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    assert isinstance(rostro_es_util(imagen, tamano_minimo=20, nitidez_minima=1.0), bool)
