"""Captura y entrenamiento en tiempo real para el proyecto Re-ID.

Este módulo resuelve el flujo práctico del laboratorio: el usuario no tiene que
copiar imágenes manualmente para Re-ID. Puede registrar a una persona desde una
cámara o video, el sistema detecta la persona, recorta torso/ropa, guarda muestras
HSV y luego entrena el SVM Re-ID cuando existan suficientes identidades.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, recortar_torso, rostro_es_util
from .datos import construir_dataset_reid, construir_dataset_rostros, guardar_imagen_bgr, guardar_metadata_csv
from .deteccion import Deteccion, DetectorPersonasYOLO, DetectorRostros
from .modelos_svm import entrenar_modelos_principales


@dataclass
class ResumenCaptura:
    """Resumen de una sesión de registro desde cámara o video."""

    identidad: str
    muestras_rostro: int
    muestras_reid: int
    carpeta_rostros: str
    carpeta_reid: str


def abrir_fuente_video(fuente: str):
    """Abre webcam, URL IP/RTSP/HTTP o archivo de video."""
    if str(fuente).isdigit():
        # Comentario clave: 0 normalmente representa la cámara principal de la PC o celular por USB.
        return cv2.VideoCapture(int(fuente))
    return cv2.VideoCapture(str(fuente))


def seleccionar_deteccion_principal(detecciones: Iterable[Deteccion]) -> Deteccion | None:
    """Escoge la detección de mayor área cuando hay varias personas."""
    detecciones = list(detecciones)
    if not detecciones:
        return None

    # Comentario clave: para registrar una identidad se usa la persona dominante frente a la cámara.
    return max(detecciones, key=lambda d: (d.caja[2] - d.caja[0]) * (d.caja[3] - d.caja[1]))


def guardar_imagen(ruta: Path, imagen: np.ndarray) -> None:
    """Guarda una imagen creando carpetas si hace falta."""
    guardar_imagen_bgr(ruta, imagen)


def registrar_metadata(ruta_csv: Path, fila: Dict[str, object]) -> None:
    """Agrega una fila de trazabilidad al CSV de captura en tiempo real."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_csv.exists()
    campos = ["sample_id", "person_id", "camera_id", "frame_number", "bbox", "tipo", "ruta", "fecha"]
    with ruta_csv.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            escritor.writeheader()
        escritor.writerow(fila)


def capturar_muestras_tiempo_real(
    configuracion: Dict[str, object],
    identidad: str,
    fuente: str = "0",
    modo_captura: str = "reid",
    muestras_objetivo: int = 40,
    intervalo_frames: int = 5,
    mostrar_ventana: bool = True,
) -> ResumenCaptura:
    """Captura muestras de rostro y/o Re-ID desde una fuente de video.

    modo_captura puede ser: reid, rostro o ambos.
    """
    rutas = configuracion.get("rutas", {})
    carpeta_rostros = Path(str(rutas.get("rostros", "datos/rostros"))) / identidad
    carpeta_reid = Path(str(rutas.get("reidentificacion", "datos/reidentificacion"))) / identidad
    ruta_metadata = Path(str(rutas.get("registros", "registros"))) / "metadata_tiempo_real.csv"

    detector_personas = DetectorPersonasYOLO(
        pesos=str(configuracion.get("yolo", {}).get("pesos", "modelos/yolov8n.pt")),
        confianza=float(configuracion.get("yolo", {}).get("confianza", 0.40)),
        tamano_imagen=int(configuracion.get("yolo", {}).get("tamano_imagen", 640)),
        dispositivo=str(configuracion.get("yolo", {}).get("dispositivo", "cpu")),
    )
    detector_rostros = DetectorRostros()
    captura = abrir_fuente_video(fuente)
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    contador_frame = 0
    muestras_reid = 0
    muestras_rostro = 0
    camara_id = "camara_01" if str(fuente).isdigit() else Path(str(fuente)).stem
    tiempo_inicio = time.time()

    print("[INFO] Registro en tiempo real iniciado. Presiona 'q' para terminar.")
    print(f"[INFO] Identidad: {identidad} | modo: {modo_captura} | fuente: {fuente}")

    while muestras_reid < muestras_objetivo or (modo_captura in {"rostro", "ambos"} and muestras_rostro < max(5, muestras_objetivo // 3)):
        ok, frame = captura.read()
        if not ok:
            break
        contador_frame += 1

        detecciones = detector_personas.detectar_personas(frame)
        principal = seleccionar_deteccion_principal(detecciones)
        salida = frame.copy()

        if principal and contador_frame % max(1, intervalo_frames) == 0:
            x1, y1, x2, y2 = principal.caja
            cv2.rectangle(salida, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if modo_captura in {"reid", "ambos"} and muestras_reid < muestras_objetivo:
                torso = recortar_torso(principal.roi)
                sample_id = f"{identidad}_{camara_id}_reid_{muestras_reid + 1:04d}"
                ruta = carpeta_reid / f"{sample_id}.jpg"
                guardar_imagen(ruta, torso)
                registrar_metadata(
                    ruta_metadata,
                    {
                        "sample_id": sample_id,
                        "person_id": identidad,
                        "camera_id": camara_id,
                        "frame_number": contador_frame,
                        "bbox": list(principal.caja),
                        "tipo": "reid_torso_hsv",
                        "ruta": str(ruta),
                        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                muestras_reid += 1

            if modo_captura in {"rostro", "ambos"}:
                rostro = detector_rostros.detectar_rostro_principal(principal.roi)
                if rostro and rostro_es_util(rostro.roi):
                    sample_id = f"{identidad}_{camara_id}_rostro_{muestras_rostro + 1:04d}"
                    ruta = carpeta_rostros / f"{sample_id}.jpg"
                    guardar_imagen(ruta, rostro.roi)
                    registrar_metadata(
                        ruta_metadata,
                        {
                            "sample_id": sample_id,
                            "person_id": identidad,
                            "camera_id": camara_id,
                            "frame_number": contador_frame,
                            "bbox": list(principal.caja),
                            "tipo": "rostro_hog",
                            "ruta": str(ruta),
                            "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )
                    muestras_rostro += 1

            texto = f"{identidad} | rostro:{muestras_rostro} reid:{muestras_reid}/{muestras_objetivo}"
            cv2.putText(salida, texto, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if mostrar_ventana:
            cv2.imshow("Registro Re-ID tiempo real", salida)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    captura.release()
    if mostrar_ventana:
        cv2.destroyAllWindows()

    duracion = max(0.01, time.time() - tiempo_inicio)
    print(f"[INFO] Captura finalizada en {duracion:.1f}s. Rostros: {muestras_rostro}, Re-ID: {muestras_reid}")
    return ResumenCaptura(
        identidad=identidad,
        muestras_rostro=muestras_rostro,
        muestras_reid=muestras_reid,
        carpeta_rostros=str(carpeta_rostros),
        carpeta_reid=str(carpeta_reid),
    )


def entrenar_desde_capturas(configuracion: Dict[str, object]) -> Dict[str, object]:
    """Entrena modelos SVM usando las capturas existentes del proyecto."""
    rutas = configuracion["rutas"]
    entrenamiento = configuracion.get("entrenamiento", {})

    max_muestras = int(entrenamiento.get("max_muestras_por_clase", 0))
    semilla = int(entrenamiento.get("semilla", 42))

    vectores_rostro, etiquetas_rostro, muestras_rostro = construir_dataset_rostros(
        rutas["rostros"],
        max_muestras_por_clase=max_muestras,
        semilla=semilla,
    )
    vectores_reid, etiquetas_reid, muestras_reid = construir_dataset_reid(
        rutas["reidentificacion"],
        max_muestras_por_clase=max_muestras,
        semilla=semilla,
    )

    # Comentario clave: se guarda metadata del entrenamiento para evidenciar qué muestras se usaron.
    guardar_metadata_csv([*muestras_rostro, *muestras_reid], Path(rutas["registros"]) / "metadata_entrenamiento.csv")

    modelos = entrenar_modelos_principales(
        vectores_rostro,
        etiquetas_rostro,
        vectores_reid,
        etiquetas_reid,
        rutas["modelos"],
        kernel=str(entrenamiento.get("kernel", "rbf")),
        probabilidad=bool(entrenamiento.get("probabilidad", True)),
        max_muestras_por_clase=max_muestras,
        semilla=semilla,
    )
    return modelos


def crear_dataset_demo(configuracion: Dict[str, object]) -> None:
    """Genera un dataset pequeño de prueba con colores sintéticos para validar ejecución."""
    rutas = configuracion.get("rutas", {})
    base_reid = Path(str(rutas.get("reidentificacion", "datos/reidentificacion")))
    rng = np.random.default_rng(42)
    personas = {
        "Danny_demo": (40, 40, 220),
        "John_demo": (40, 180, 40),
        "Matias_demo": (220, 80, 40),
    }

    for persona, color_bgr in personas.items():
        carpeta = base_reid / persona
        carpeta.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            imagen = np.zeros((256, 128, 3), dtype=np.uint8)
            ruido = rng.normal(0, 10, imagen.shape).astype(np.int16)
            color = np.array(color_bgr, dtype=np.int16).reshape(1, 1, 3)
            imagen[:] = np.clip(color + ruido, 0, 255).astype(np.uint8)
            # Comentario clave: estas muestras solo sirven para comprobar que el pipeline entrena y predice.
            cv2.rectangle(imagen, (20, 40), (108, 220), tuple(int(c) for c in color_bgr), -1)
            cv2.imwrite(str(carpeta / f"demo_{i + 1:03d}.jpg"), imagen)

    print(f"[INFO] Dataset demo creado en: {base_reid}")
