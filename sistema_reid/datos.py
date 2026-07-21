"""Carga de datasets organizados por identidad."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, recortar_torso
from .deteccion import DetectorRostros

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MuestraImagen:
    """Información mínima de una imagen dentro del dataset."""

    ruta: Path
    identidad: str
    tipo: str


def listar_imagenes_por_identidad(carpeta_base: str | Path, tipo: str) -> List[MuestraImagen]:
    """Lista imágenes ubicadas en subcarpetas por identidad."""
    carpeta = Path(carpeta_base)
    muestras: List[MuestraImagen] = []
    if not carpeta.exists():
        return muestras

    for carpeta_identidad in sorted(carpeta.iterdir()):
        if not carpeta_identidad.is_dir():
            continue
        identidad = carpeta_identidad.name
        for ruta in sorted(carpeta_identidad.rglob("*")):
            # Comentario clave: solo se aceptan archivos de imagen para evitar leer .gitkeep u otros archivos.
            if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                muestras.append(MuestraImagen(ruta=ruta, identidad=identidad, tipo=tipo))
    return muestras


def cargar_imagen_bgr(ruta: str | Path) -> np.ndarray:
    """Lee una imagen desde disco en formato BGR."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")

    # Comentario clave: OpenCV trabaja naturalmente con BGR; se mantiene ese formato en todo el proyecto.
    return imagen


def obtener_roi_rostro_entrenamiento(imagen: np.ndarray, detector_rostros: DetectorRostros) -> np.ndarray:
    """Devuelve el rostro principal para entrenar; si no aparece, usa la imagen original."""
    rostros = detector_rostros.detectar_rostros(imagen, tamano_minimo=40)
    if rostros:
        return rostros[0].roi
    return imagen


def construir_dataset_rostros(carpeta_rostros: str | Path) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HoG y etiquetas desde datos/rostros/<identidad>."""
    muestras = listar_imagenes_por_identidad(carpeta_rostros, tipo="rostro")
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    detector_rostros = DetectorRostros()

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        # Comentario clave: aquí se aplica HoG porque estas carpetas deben contener rostros visibles.
        roi_rostro = obtener_roi_rostro_entrenamiento(imagen, detector_rostros)
        vectores.append(extraer_hog_rostro(roi_rostro))
        etiquetas.append(muestra.identidad)

    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras


def construir_dataset_reid(carpeta_reid: str | Path) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HSV y etiquetas desde datos/reidentificacion/<identidad>."""
    muestras = listar_imagenes_por_identidad(carpeta_reid, tipo="reidentificacion")
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        # Comentario clave: las capturas Re-ID ya se guardan como torso/ropa desde la cámara;
        # si el usuario pone cuerpo completo, HSV sigue funcionando como descriptor soft-biométrico.
        vectores.append(extraer_histograma_hsv(imagen))
        etiquetas.append(muestra.identidad)

    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras


def guardar_metadata_csv(muestras: Sequence[MuestraImagen], ruta_salida: str | Path) -> None:
    """Guarda un CSV simple para mantener trazabilidad de las imágenes usadas."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["ruta", "identidad", "tipo"])
        escritor.writeheader()
        for muestra in muestras:
            # Comentario clave: este registro ayuda a defender qué imágenes alimentaron cada entrenamiento.
            escritor.writerow({"ruta": str(muestra.ruta), "identidad": muestra.identidad, "tipo": muestra.tipo})


def validar_dataset_minimo(etiquetas: Sequence[str], minimo_por_clase: int = 2) -> None:
    """Verifica que existan suficientes imágenes por identidad antes de entrenar."""
    conteo = {etiqueta: list(etiquetas).count(etiqueta) for etiqueta in set(etiquetas)}
    clases_insuficientes = [clase for clase, total in conteo.items() if total < minimo_por_clase]

    # Comentario clave: SVM necesita más de una muestra por clase para aprender una frontera útil.
    if len(conteo) < 2:
        raise ValueError("Se necesitan al menos dos identidades para entrenar SVM.")
    if clases_insuficientes:
        raise ValueError(f"Faltan imágenes en estas identidades: {clases_insuficientes}")
