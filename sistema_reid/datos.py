"""Carga de datasets organizados por identidad."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, extraer_hog_rostro
from .deteccion import DetectorRostros

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MuestraImagen:
    """Informacion minima de una imagen dentro del dataset."""

    ruta: Path
    identidad: str
    tipo: str


def listar_imagenes_por_identidad(carpeta_base: str | Path, tipo: str) -> List[MuestraImagen]:
    """Lista imagenes ubicadas en subcarpetas por identidad."""
    carpeta = Path(carpeta_base)
    muestras: List[MuestraImagen] = []
    if not carpeta.exists():
        return muestras

    for carpeta_identidad in sorted(carpeta.iterdir()):
        if not carpeta_identidad.is_dir():
            continue
        identidad = carpeta_identidad.name
        for ruta in sorted(carpeta_identidad.rglob("*")):
            if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                muestras.append(MuestraImagen(ruta=ruta, identidad=identidad, tipo=tipo))
    return muestras


def limitar_muestras_por_identidad(
    muestras: Sequence[MuestraImagen],
    maximo_por_identidad: Optional[int],
    semilla: int = 42,
) -> List[MuestraImagen]:
    """Limita fotos por identidad antes de extraer descriptores costosos."""
    if maximo_por_identidad is None or maximo_por_identidad <= 0:
        return list(muestras)

    rng = np.random.default_rng(semilla)
    seleccionadas: List[MuestraImagen] = []
    identidades = sorted({muestra.identidad for muestra in muestras})

    for identidad in identidades:
        grupo = [muestra for muestra in muestras if muestra.identidad == identidad]
        if len(grupo) > maximo_por_identidad:
            indices = sorted(rng.choice(len(grupo), size=maximo_por_identidad, replace=False).tolist())
            grupo = [grupo[indice] for indice in indices]
        seleccionadas.extend(grupo)

    return seleccionadas


def cargar_imagen_bgr(ruta: str | Path) -> np.ndarray:
    """Lee una imagen desde disco en formato BGR, incluso con rutas Unicode en Windows."""
    ruta = Path(ruta)
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return imagen


def guardar_imagen_bgr(ruta: str | Path, imagen: np.ndarray) -> None:
    """Guarda una imagen BGR soportando rutas Unicode en Windows."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    extension = ruta.suffix or ".jpg"
    ok, codificada = cv2.imencode(extension, imagen)
    if not ok:
        raise RuntimeError(f"No se pudo codificar la imagen: {ruta}")
    codificada.tofile(str(ruta))


def obtener_roi_rostro_entrenamiento(imagen: np.ndarray, detector_rostros: DetectorRostros) -> np.ndarray | None:
    """Devuelve solo un ROI de rostro para entrenar identificacion facial."""
    rostros = detector_rostros.detectar_rostros(imagen, tamano_minimo=40)
    if rostros:
        return rostros[0].roi
    return None


def construir_dataset_rostros(
    carpeta_rostros: str | Path,
    max_muestras_por_clase: Optional[int] = None,
    semilla: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HoG y etiquetas desde datos/rostros/<identidad>."""
    muestras = listar_imagenes_por_identidad(carpeta_rostros, tipo="rostro")
    muestras = limitar_muestras_por_identidad(muestras, max_muestras_por_clase, semilla)
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    muestras_usadas: List[MuestraImagen] = []
    omitidas = 0
    detector_rostros = DetectorRostros()

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        roi_rostro = obtener_roi_rostro_entrenamiento(imagen, detector_rostros)
        if roi_rostro is None:
            omitidas += 1
            continue
        vectores.append(extraer_hog_rostro(roi_rostro))
        etiquetas.append(muestra.identidad)
        muestras_usadas.append(muestra)

    if omitidas:
        print(f"[AVISO] Rostros omitidos por no detectar ROI facial: {omitidas}")
    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras_usadas


def construir_dataset_reid(
    carpeta_reid: str | Path,
    max_muestras_por_clase: Optional[int] = None,
    semilla: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HSV y etiquetas desde datos/reidentificacion/<identidad>."""
    muestras = listar_imagenes_por_identidad(carpeta_reid, tipo="reidentificacion")
    muestras = limitar_muestras_por_identidad(muestras, max_muestras_por_clase, semilla)
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        vectores.append(extraer_histograma_hsv(imagen))
        etiquetas.append(muestra.identidad)

    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras


def guardar_metadata_csv(muestras: Sequence[MuestraImagen], ruta_salida: str | Path) -> None:
    """Guarda un CSV simple para mantener trazabilidad de las imagenes usadas."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["ruta", "identidad", "tipo"])
        escritor.writeheader()
        for muestra in muestras:
            escritor.writerow({"ruta": str(muestra.ruta), "identidad": muestra.identidad, "tipo": muestra.tipo})


def validar_dataset_minimo(etiquetas: Sequence[str], minimo_por_clase: int = 2) -> None:
    """Verifica que existan suficientes imagenes por identidad antes de entrenar."""
    conteo = {etiqueta: list(etiquetas).count(etiqueta) for etiqueta in set(etiquetas)}
    clases_insuficientes = [clase for clase, total in conteo.items() if total < minimo_por_clase]

    if len(conteo) < 2:
        raise ValueError("Se necesitan al menos dos identidades para entrenar SVM.")
    if clases_insuficientes:
        raise ValueError(f"Faltan imagenes en estas identidades: {clases_insuficientes}")
