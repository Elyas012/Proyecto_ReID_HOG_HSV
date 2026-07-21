"""Extracción de características HoG y HSV para identificación y Re-ID."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np
from skimage.feature import hog

Caja = Tuple[int, int, int, int]


def asegurar_bgr(imagen: np.ndarray) -> np.ndarray:
    """Garantiza que la imagen tenga tres canales BGR para OpenCV."""
    if imagen is None:
        raise ValueError("La imagen recibida es None.")

    # Comentario clave: algunas imágenes llegan en gris; se convierten para unificar el flujo.
    if len(imagen.shape) == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    if imagen.shape[2] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2BGR)
    return imagen


def redimensionar_roi(roi: np.ndarray, tamano: Iterable[int]) -> np.ndarray:
    """Redimensiona una región de interés manteniendo un tamaño fijo para el modelo."""
    ancho, alto = [int(valor) for valor in tamano]
    if roi is None or roi.size == 0:
        raise ValueError("El ROI está vacío y no puede redimensionarse.")

    # Comentario clave: todos los vectores deben salir con la misma dimensión para poder entrenar SVM.
    return cv2.resize(roi, (ancho, alto), interpolation=cv2.INTER_AREA)


def convertir_a_gris(imagen: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR o BGRA a escala de grises."""
    imagen = asegurar_bgr(imagen)
    # Comentario clave: HoG trabaja mejor sobre intensidad, por eso se usa escala de grises.
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def extraer_hog_rostro(
    roi_rostro: np.ndarray,
    tamano: Tuple[int, int] = (96, 96),
    orientaciones: int = 9,
    pixeles_por_celda: Tuple[int, int] = (8, 8),
    celdas_por_bloque: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extrae el descriptor HoG de un rostro visible y normalizado."""
    rostro = redimensionar_roi(roi_rostro, tamano)
    rostro_gris = convertir_a_gris(rostro)

    # Comentario clave: HoG resume bordes y gradientes del rostro para alimentar el SVM facial.
    vector = hog(
        rostro_gris,
        orientations=orientaciones,
        pixels_per_cell=pixeles_por_celda,
        cells_per_block=celdas_por_bloque,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return vector.astype("float32")


def extraer_histograma_hsv(
    roi_torso: np.ndarray,
    tamano: Tuple[int, int] = (128, 256),
    bins: Tuple[int, int, int] = (16, 16, 8),
) -> np.ndarray:
    """Extrae un histograma HSV del torso/ropa para re-identificación sin rostro."""
    torso = redimensionar_roi(asegurar_bgr(roi_torso), tamano)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Comentario clave: HSV representa la distribución de color de la ropa, útil cuando no se ve la cara.
    histograma = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    histograma = cv2.normalize(histograma, histograma).flatten()
    return histograma.astype("float32")


def recortar_caja(imagen: np.ndarray, caja: Caja) -> np.ndarray:
    """Recorta una imagen usando una caja (x1, y1, x2, y2) y corrige límites inválidos."""
    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in caja]

    # Comentario clave: se limita la caja para que nunca intente recortar fuera del frame.
    x1 = max(0, min(x1, ancho - 1))
    x2 = max(0, min(x2, ancho))
    y1 = max(0, min(y1, alto - 1))
    y2 = max(0, min(y2, alto))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=imagen.dtype)
    return imagen[y1:y2, x1:x2]


def recortar_torso(
    roi_persona: np.ndarray,
    porcentaje_superior: float = 0.25,
    porcentaje_inferior: float = 0.90,
) -> np.ndarray:
    """Obtiene la zona aproximada de torso/ropa dentro del ROI de una persona."""
    if roi_persona is None or roi_persona.size == 0:
        raise ValueError("El ROI de persona está vacío.")

    alto, ancho = roi_persona.shape[:2]
    y1 = int(alto * porcentaje_superior)
    y2 = int(alto * porcentaje_inferior)

    # Comentario clave: se evita la parte baja/piernas para priorizar color de torso y camiseta.
    return roi_persona[y1:y2, 0:ancho]


def calcular_nitidez(imagen: np.ndarray) -> float:
    """Calcula la nitidez mediante la varianza del Laplaciano."""
    gris = convertir_a_gris(imagen)
    # Comentario clave: valores bajos indican imagen borrosa, por lo que el rostro no debería usarse.
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def rostro_es_util(roi_rostro: np.ndarray, tamano_minimo: int = 40, nitidez_minima: float = 60.0) -> bool:
    """Valida si un rostro tiene tamaño y nitidez suficientes para usar HoG + SVM."""
    if roi_rostro is None or roi_rostro.size == 0:
        return False

    alto, ancho = roi_rostro.shape[:2]
    if min(alto, ancho) < tamano_minimo:
        return False

    # Comentario clave: el sistema solo usa reconocimiento facial si el rostro es confiable.
    return calcular_nitidez(roi_rostro) >= nitidez_minima
