"""Pruebas básicas de extracción de características."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sistema_reid.caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, recortar_torso, rostro_es_util
from sistema_reid.modelos_svm import ArtefactosSVM, contar_muestras_por_clase, limitar_muestras_por_clase, predecir_con_confianza


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
    assert vector.size == (16 * 16 * 8 + 6) * 4


def test_rostro_es_util_devuelve_booleano():
    imagen = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    assert isinstance(rostro_es_util(imagen, tamano_minimo=20, nitidez_minima=1.0), bool)


def test_limitar_muestras_por_clase_reduce_clase_dominante():
    vectores = np.arange(18, dtype="float32").reshape(9, 2)
    etiquetas = np.array(["A", "A", "A", "A", "B", "B", "C", "C", "C"])

    _, etiquetas_balanceadas = limitar_muestras_por_clase(vectores, etiquetas, maximo_por_clase=2, semilla=1)

    assert contar_muestras_por_clase(etiquetas_balanceadas) == {"A": 2, "B": 2, "C": 2}


def test_predecir_usa_predict_como_identidad():
    class EscaladorDummy:
        def transform(self, vector):
            return vector

    class ModeloDummy:
        classes_ = np.array(["Elias", "John"])

        def predict(self, vector):
            return np.array(["John"])

        def predict_proba(self, vector):
            return np.array([[0.90, 0.10]])

    artefactos = ArtefactosSVM(ModeloDummy(), EscaladorDummy(), ModeloDummy.classes_, "rostro_hog_svm")

    identidad, score, ranking = predecir_con_confianza(artefactos, np.array([1.0, 2.0]))

    assert identidad == "John"
    assert score == ranking["John"]
