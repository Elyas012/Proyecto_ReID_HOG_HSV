"""Extracción de características HoG y HSV para identificación y Re-ID."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

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


def _extraer_histograma_hsv_global(
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


def _limitar_float(valor: object, minimo: float, maximo: float, defecto: float) -> float:
    """Convierte y limita un valor numerico de configuracion."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        numero = defecto
    return max(minimo, min(maximo, numero))


def _normalizar_tupla_enteros(valor: object, defecto: Tuple[int, ...]) -> Tuple[int, ...]:
    """Normaliza listas del YAML usadas como tamanos o bins."""
    if not isinstance(valor, (list, tuple)) or len(valor) != len(defecto):
        return defecto
    try:
        return tuple(max(1, int(v)) for v in valor)
    except (TypeError, ValueError):
        return defecto


def _limitar_int(valor: object, minimo: int, maximo: int, defecto: int) -> int:
    """Convierte y limita un entero de configuracion."""
    try:
        numero = int(valor)
    except (TypeError, ValueError):
        numero = defecto
    return max(minimo, min(maximo, numero))


def parametros_hsv_desde_config(configuracion: Dict[str, object]) -> Dict[str, object]:
    """Extrae del YAML los parametros del descriptor HSV de Re-ID."""
    caracteristicas = configuracion.get("caracteristicas", {}) if isinstance(configuracion, dict) else {}
    if not isinstance(caracteristicas, dict):
        caracteristicas = {}

    return {
        "tamano": _normalizar_tupla_enteros(caracteristicas.get("tamano_torso"), (128, 256)),
        "bins": _normalizar_tupla_enteros(caracteristicas.get("bins_hsv"), (16, 16, 8)),
        "bandas_horizontales": _limitar_int(caracteristicas.get("reid_bandas_horizontales", 3), 1, 8, 3),
        "usar_global": bool(caracteristicas.get("reid_histograma_global", True)),
        "incluir_momentos": bool(caracteristicas.get("reid_incluir_momentos", True)),
        "recorte_lateral": _limitar_float(caracteristicas.get("reid_recorte_lateral", 0.08), 0.0, 0.35, 0.08),
        "recorte_superior": _limitar_float(caracteristicas.get("reid_recorte_superior", 0.0), 0.0, 0.35, 0.0),
        "recorte_inferior": _limitar_float(caracteristicas.get("reid_recorte_inferior", 0.0), 0.0, 0.35, 0.0),
    }


def dimension_histograma_hsv(parametros: Dict[str, object]) -> int:
    """Calcula la dimension esperada del descriptor HSV espacial."""
    bins = _normalizar_tupla_enteros(parametros.get("bins"), (16, 16, 8))
    bandas = _limitar_int(parametros.get("bandas_horizontales", 3), 1, 8, 3)
    usar_global = bool(parametros.get("usar_global", True))
    incluir_momentos = bool(parametros.get("incluir_momentos", True))
    regiones = (1 if usar_global or bandas == 1 else 0) + (bandas if bandas > 1 else 0)
    dimension_region = int(np.prod(bins)) + (6 if incluir_momentos else 0)
    return regiones * dimension_region


def recortar_margenes_reid(
    imagen: np.ndarray,
    recorte_lateral: float = 0.08,
    recorte_superior: float = 0.0,
    recorte_inferior: float = 0.0,
) -> np.ndarray:
    """Reduce borde/fondo dentro del ROI para que Re-ID mire mas la ropa."""
    alto, ancho = imagen.shape[:2]
    x1 = int(ancho * recorte_lateral)
    x2 = int(ancho * (1.0 - recorte_lateral))
    y1 = int(alto * recorte_superior)
    y2 = int(alto * (1.0 - recorte_inferior))
    if x2 <= x1 or y2 <= y1:
        return imagen
    return imagen[y1:y2, x1:x2]


def _histograma_hsv_region(region_bgr: np.ndarray, bins: Tuple[int, int, int]) -> np.ndarray:
    """Calcula un histograma HSV normalizado para una region."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    histograma = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    histograma = cv2.normalize(histograma, histograma, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    return histograma.astype("float32")


def _momentos_hsv_region(region_bgr: np.ndarray) -> np.ndarray:
    """Resume promedio y variacion HSV para reforzar el histograma."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    medias, desvios = cv2.meanStdDev(hsv)
    escala = np.asarray([180.0, 256.0, 256.0], dtype="float32")
    vector = np.concatenate([(medias.flatten() / escala), (desvios.flatten() / escala)])
    return vector.astype("float32")


def extraer_histograma_hsv(
    roi_torso: np.ndarray,
    tamano: Tuple[int, int] = (128, 256),
    bins: Tuple[int, int, int] = (16, 16, 8),
    bandas_horizontales: int = 3,
    usar_global: bool = True,
    incluir_momentos: bool = True,
    recorte_lateral: float = 0.08,
    recorte_superior: float = 0.0,
    recorte_inferior: float = 0.0,
) -> np.ndarray:
    """Extrae HSV espacial del torso/ropa para re-identificacion sin rostro."""
    torso = redimensionar_roi(asegurar_bgr(roi_torso), tamano)
    torso = recortar_margenes_reid(torso, recorte_lateral, recorte_superior, recorte_inferior)
    bandas_horizontales = max(1, min(8, int(bandas_horizontales)))

    regiones = []
    if usar_global or bandas_horizontales == 1:
        regiones.append(torso)

    if bandas_horizontales > 1:
        alto = torso.shape[0]
        for indice in range(bandas_horizontales):
            y1 = int(indice * alto / bandas_horizontales)
            y2 = int((indice + 1) * alto / bandas_horizontales)
            if y2 > y1:
                regiones.append(torso[y1:y2, :])

    # Comentario clave: las bandas hacen que dos personas con colores parecidos no queden tan iguales.
    partes = []
    for region in regiones:
        partes.append(_histograma_hsv_region(region, bins))
        if incluir_momentos:
            partes.append(_momentos_hsv_region(region))

    return np.concatenate(partes).astype("float32")


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
