"""Captura y entrenamiento en tiempo real para el proyecto Re-ID.

Este mÃ³dulo resuelve el flujo prÃ¡ctico del laboratorio: el usuario no tiene que
copiar imÃ¡genes manualmente para Re-ID. Puede registrar a una persona desde una
cÃ¡mara o video, el sistema detecta la persona completa, guarda muestras
HSV y luego entrena el SVM Re-ID cuando existan suficientes identidades.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, parametros_hog_desde_config, parametros_hsv_desde_config, rostro_es_util
from .configuracion import resolver_limites_entrenamiento
from .datos import (
    construir_augmentation_reid,
    construir_dataset_reid,
    construir_dataset_rostros,
    guardar_imagen_bgr,
    guardar_metadata_csv,
    limitar_imagenes_recientes_por_identidad,
)
from .deteccion import Deteccion, DetectorPersonasYOLO, DetectorRostros
from .evaluacion import evaluar_modelo
from .modelos_svm import dividir_indices_validacion, entrenar_modelos_principales


@dataclass
class ResumenCaptura:
    """Resumen de una sesiÃ³n de registro desde cÃ¡mara o video."""

    identidad: str
    muestras_rostro: int
    muestras_reid: int
    carpeta_rostros: str
    carpeta_reid: str


def abrir_fuente_video(fuente: str):
    """Abre webcam, URL IP/RTSP/HTTP o archivo de video."""
    if str(fuente).isdigit():
        # Comentario clave: 0 normalmente representa la cÃ¡mara principal de la PC o celular por USB.
        return cv2.VideoCapture(int(fuente))
    return cv2.VideoCapture(str(fuente))


def seleccionar_deteccion_principal(detecciones: Iterable[Deteccion]) -> Deteccion | None:
    """Escoge la deteccion de mayor area cuando hay varias personas."""
    detecciones = list(detecciones)
    if not detecciones:
        return None

    # Comentario clave: para registrar una identidad se usa la persona dominante frente a la cÃ¡mara.
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


def contar_muestras(muestras: Iterable[object]) -> Dict[str, int]:
    """Cuenta muestras por identidad para reportar el entrenamiento usado."""
    conteo = Counter(str(muestra.identidad) for muestra in muestras)
    return {identidad: conteo[identidad] for identidad in sorted(conteo)}


def preparar_augmentation_reidF(configuracion: Dict[str, object]) -> Dict[str, object]:
    """Prepara la configuracion de augmentation para las capturas automaticas Re-ID."""
    augmentation = dict(configuracion.get("augmentation_reidF", {}) or {})
    if not bool(augmentation.get("activo", False)):
        return {"activo": False}

    aprendizaje = configuracion.get("aprendizaje_reid_en_vivo", {})
    augmentation.setdefault(
        "max_imagenes_por_identidad",
        int(aprendizaje.get("max_imagenes_carpeta_por_identidad", aprendizaje.get("max_muestras_vivas_por_identidad", 50))),
    )
    return augmentation


def dividir_dataset_validacion(
    vectores: np.ndarray,
    etiquetas: np.ndarray,
    muestras: List[object],
    proporcion_validacion: float,
    semilla: int,
    validacion_por_clase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[object], np.ndarray, np.ndarray, List[object]]:
    """Divide un dataset por identidad y conserva muestras suficientes para entrenar."""
    if vectores is None or etiquetas is None or len(vectores) == 0 or proporcion_validacion <= 0:
        return vectores, etiquetas, muestras, np.asarray([], dtype="float32"), np.asarray([]), []

    etiquetas = np.asarray(etiquetas)
    indices_entrenamiento, indices_validacion = dividir_indices_validacion(
        etiquetas,
        proporcion_validacion,
        semilla=semilla,
        validacion_por_clase=validacion_por_clase,
    )
    if len(indices_validacion) == 0:
        return vectores, etiquetas, muestras, np.asarray([], dtype="float32"), np.asarray([]), []

    muestras_entrenamiento = [muestras[indice] for indice in indices_entrenamiento]
    muestras_validacion = [muestras[indice] for indice in indices_validacion]
    return (
        vectores[indices_entrenamiento],
        etiquetas[indices_entrenamiento],
        muestras_entrenamiento,
        vectores[indices_validacion],
        etiquetas[indices_validacion],
        muestras_validacion,
    )


def guardar_resumen_entrenamiento(
    configuracion: Dict[str, object],
    muestras_rostro: Iterable[object],
    muestras_reid: Iterable[object],
    modelos: Dict[str, object],
    muestras_rostro_validacion: Iterable[object] | None = None,
    muestras_reid_validacion: Iterable[object] | None = None,
    metricas_validacion: Dict[str, object] | None = None,
) -> None:
    """Guarda un resumen ligero del ultimo entrenamiento para el panel."""
    rutas = configuracion["rutas"]
    entrenamiento = configuracion.get("entrenamiento", {})
    caracteristicas = configuracion.get("caracteristicas", {})
    augmentation_reidF = configuracion.get("augmentation_reidF", {})
    limites = resolver_limites_entrenamiento(configuracion)
    ruta_resumen = Path(rutas["reportes"]) / "resumen_entrenamiento.json"
    ruta_resumen.parent.mkdir(parents=True, exist_ok=True)
    resumen = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_muestras_por_clase": limites["max_muestras_por_clase"],
        "max_muestras_rostro_por_clase": limites["max_muestras_rostro_por_clase"],
        "min_muestras_reid_por_clase": limites["min_muestras_reid_por_clase"],
        "max_muestras_reid_por_clase": limites["max_muestras_reid_por_clase"],
        "kernel_rostro": str(entrenamiento.get("kernel_rostro", entrenamiento.get("kernel", "rbf"))),
        "kernel_reid": str(entrenamiento.get("kernel_reid", entrenamiento.get("kernel", "rbf"))),
        "validacion": float(entrenamiento.get("validacion", 0.20)),
        "validacion_por_clase": bool(entrenamiento.get("validacion_por_clase", True)),
        "omitir_rostros_sin_roi": bool(entrenamiento.get("omitir_rostros_sin_roi", True)),
        "filtrar_rostros_baja_calidad": bool(entrenamiento.get("filtrar_rostros_baja_calidad", True)),
        "descriptor_reid": str(caracteristicas.get("descriptor_reid", "hsv_espacial")),
        "parametros_hsv_reid": parametros_hsv_desde_config(configuracion),
        "augmentation_reidF": augmentation_reidF,
        "conteo_rostro_entrenamiento": contar_muestras(muestras_rostro),
        "conteo_rostro_validacion": contar_muestras(muestras_rostro_validacion or []),
        "conteo_reid_entrenamiento": contar_muestras(muestras_reid),
        "conteo_reid_validacion": contar_muestras(muestras_reid_validacion or []),
        "conteo_rostro_usado": contar_muestras(muestras_rostro),
        "conteo_reid_usado": contar_muestras(muestras_reid),
        "metricas_validacion": metricas_validacion or {},
        "modelos_entrenados": sorted(modelos.keys()),
    }
    with ruta_resumen.open("w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)


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
    aprendizaje = configuracion.get("aprendizaje_reid_en_vivo", {})
    carpeta_rostros = Path(str(rutas.get("rostros", "datos/rostros"))) / identidad
    carpeta_reid = Path(str(rutas.get("reidentificacionF", "datos/reidentificacionF"))) / identidad
    ruta_metadata = Path(str(rutas.get("registros", "registros"))) / "metadata_tiempo_real.csv"
    max_imagenes_reid = int(
        aprendizaje.get("max_imagenes_carpeta_por_identidad", aprendizaje.get("max_muestras_vivas_por_identidad", 50))
    )

    detector_personas = DetectorPersonasYOLO(
        pesos=str(configuracion.get("yolo", {}).get("pesos", "modelos/yolov8n.pt")),
        confianza=float(configuracion.get("yolo", {}).get("confianza", 0.40)),
        tamano_imagen=int(configuracion.get("yolo", {}).get("tamano_imagen", 640)),
        dispositivo=str(configuracion.get("yolo", {}).get("dispositivo", "cpu")),
    )
    detector_rostros = DetectorRostros.desde_config(configuracion)
    captura = abrir_fuente_video(fuente)
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    contador_frame = 0
    muestras_reid = 0
    muestras_rostro = 0
    objetivo_reid = muestras_objetivo if modo_captura in {"reid", "ambos"} else 0
    objetivo_rostro = muestras_objetivo if modo_captura == "rostro" else (max(5, muestras_objetivo // 3) if modo_captura == "ambos" else 0)
    camara_id = "camara_01" if str(fuente).isdigit() else Path(str(fuente)).stem
    tiempo_inicio = time.time()

    print("[INFO] Registro en tiempo real iniciado. Presiona 'q' para terminar.")
    print(f"[INFO] Identidad: {identidad} | modo: {modo_captura} | fuente: {fuente}")

    while muestras_reid < objetivo_reid or muestras_rostro < objetivo_rostro:
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

            if modo_captura in {"reid", "ambos"} and muestras_reid < objetivo_reid:
                cuerpo = principal.roi
                sample_id = f"{identidad}_{camara_id}_reid_{muestras_reid + 1:04d}"
                ruta = carpeta_reid / f"{sample_id}.jpg"
                guardar_imagen(ruta, cuerpo)
                limitar_imagenes_recientes_por_identidad(carpeta_reid, max_imagenes_reid)
                registrar_metadata(
                    ruta_metadata,
                    {
                        "sample_id": sample_id,
                        "person_id": identidad,
                        "camera_id": camara_id,
                        "frame_number": contador_frame,
                        "bbox": list(principal.caja),
                        "tipo": "reid_cuerpo_hsv_3_bandas",
                        "ruta": str(ruta),
                        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                muestras_reid += 1

            if modo_captura in {"rostro", "ambos"} and muestras_rostro < objetivo_rostro:
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

            texto = f"{identidad} | rostro:{muestras_rostro}/{objetivo_rostro} reid:{muestras_reid}/{objetivo_reid}"
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
    limites = resolver_limites_entrenamiento(configuracion)

    max_muestras = limites["max_muestras_por_clase"]
    max_muestras_rostro = limites["max_muestras_rostro_por_clase"]
    max_muestras_reid = limites["max_muestras_reid_por_clase"]
    min_muestras_reid = limites["min_muestras_reid_por_clase"]
    semilla = int(entrenamiento.get("semilla", 42))
    omitir_sin_roi = bool(entrenamiento.get("omitir_rostros_sin_roi", True))
    filtrar_rostros_baja_calidad = bool(entrenamiento.get("filtrar_rostros_baja_calidad", True))
    proporcion_validacion = float(entrenamiento.get("validacion", 0.20))
    validacion_por_clase = bool(entrenamiento.get("validacion_por_clase", True))
    parametros_hsv = parametros_hsv_desde_config(configuracion)
    parametros_hog = parametros_hog_desde_config(configuracion)
    augmentation_reidF = preparar_augmentation_reidF(configuracion)
    umbrales = configuracion.get("umbrales", {})

    vectores_rostro, etiquetas_rostro, muestras_rostro = construir_dataset_rostros(
        rutas["rostros"],
        max_muestras_por_clase=max_muestras_rostro,
        semilla=semilla,
        omitir_sin_roi=omitir_sin_roi,
        filtrar_baja_calidad=filtrar_rostros_baja_calidad,
        tamano_minimo=int(umbrales.get("tamano_minimo_rostro", 40)),
        nitidez_minima=float(umbrales.get("nitidez_minima", 60.0)),
        parametros_hog=parametros_hog,
        configuracion=configuracion,
    )
    vectores_reid = np.asarray([], dtype="float32")
    etiquetas_reid = np.asarray([])
    muestras_reid: List[object] = []
    ruta_reid_vivo = rutas.get("reidentificacionF")
    if ruta_reid_vivo and Path(str(ruta_reid_vivo)).exists():
        vectores_reid_vivo, etiquetas_reid_vivo, muestras_reid_vivo = construir_dataset_reid(
            ruta_reid_vivo,
            max_muestras_por_clase=max_muestras_reid,
            min_muestras_por_clase=min_muestras_reid,
            semilla=semilla,
            parametros_hsv=parametros_hsv,
            tipo="reidentificacionF",
            incluir_aumentadas=False,
        )
        if len(vectores_reid_vivo) > 0:
            if len(vectores_reid) > 0:
                vectores_reid = np.concatenate([vectores_reid, vectores_reid_vivo], axis=0)
                etiquetas_reid = np.concatenate([etiquetas_reid, etiquetas_reid_vivo], axis=0)
                muestras_reid = [*muestras_reid, *muestras_reid_vivo]
            else:
                vectores_reid, etiquetas_reid, muestras_reid = vectores_reid_vivo, etiquetas_reid_vivo, muestras_reid_vivo
            print(f"[INFO] Re-ID vivo en carpeta F agregado al entrenamiento: {contar_muestras(muestras_reid_vivo)}")

    # Comentario clave: se guarda metadata del entrenamiento para evidenciar quÃ© muestras se usaron.
    (
        vectores_rostro_train,
        etiquetas_rostro_train,
        muestras_rostro_train,
        vectores_rostro_val,
        etiquetas_rostro_val,
        muestras_rostro_val,
    ) = dividir_dataset_validacion(
        vectores_rostro,
        etiquetas_rostro,
        muestras_rostro,
        proporcion_validacion,
        semilla,
        validacion_por_clase=validacion_por_clase,
    )
    (
        vectores_reid_train,
        etiquetas_reid_train,
        muestras_reid_train,
        vectores_reid_val,
        etiquetas_reid_val,
        muestras_reid_val,
    ) = dividir_dataset_validacion(
        vectores_reid,
        etiquetas_reid,
        muestras_reid,
        proporcion_validacion,
        semilla,
        validacion_por_clase=validacion_por_clase,
    )

    muestras_reidF_train = [muestra for muestra in muestras_reid_train if getattr(muestra, "tipo", "") == "reidentificacionF"]
    vectores_reid_aug, etiquetas_reid_aug, muestras_reid_aug = construir_augmentation_reid(
        muestras_reidF_train,
        semilla=semilla,
        parametros_hsv=parametros_hsv,
        augmentacion=augmentation_reidF,
    )
    if len(vectores_reid_aug) > 0:
        vectores_reid_train = np.concatenate([vectores_reid_train, vectores_reid_aug], axis=0)
        etiquetas_reid_train = np.concatenate([etiquetas_reid_train, etiquetas_reid_aug], axis=0)
        muestras_reid_train = [*muestras_reid_train, *muestras_reid_aug]

    print(f"[INFO] Split rostro train/val: {contar_muestras(muestras_rostro_train)} / {contar_muestras(muestras_rostro_val)}")
    print(f"[INFO] Split Re-ID train/val: {contar_muestras(muestras_reid_train)} / {contar_muestras(muestras_reid_val)}")

    guardar_metadata_csv([*muestras_rostro_train, *muestras_reid_train], Path(rutas["registros"]) / "metadata_entrenamiento.csv")
    guardar_metadata_csv([*muestras_rostro_val, *muestras_reid_val], Path(rutas["registros"]) / "metadata_validacion.csv")

    modelos = entrenar_modelos_principales(
        vectores_rostro_train,
        etiquetas_rostro_train,
        vectores_reid_train,
        etiquetas_reid_train,
        rutas["modelos"],
        kernel=str(entrenamiento.get("kernel", "rbf")),
        kernel_rostro=str(entrenamiento.get("kernel_rostro", entrenamiento.get("kernel", "rbf"))),
        kernel_reid=str(entrenamiento.get("kernel_reid", entrenamiento.get("kernel", "rbf"))),
        probabilidad=bool(entrenamiento.get("probabilidad", True)),
        max_muestras_por_clase=max_muestras,
        max_muestras_rostro_por_clase=max_muestras_rostro,
        max_muestras_reid_por_clase=max_muestras_reid,
        semilla=semilla,
    )
    metricas_validacion: Dict[str, object] = {}
    if "svm_rostro" in modelos and len(vectores_rostro_val) > 0:
        metricas_validacion["rostro"] = evaluar_modelo(modelos["svm_rostro"], vectores_rostro_val, etiquetas_rostro_val)
        print(
            "[INFO] Validacion rostro: "
            f"accuracy={metricas_validacion['rostro']['accuracy']:.3f} "
            f"f1={metricas_validacion['rostro']['f1_macro']:.3f}"
        )
    if "svm_reidentificacion" in modelos and len(vectores_reid_val) > 0:
        metricas_validacion["reid"] = evaluar_modelo(modelos["svm_reidentificacion"], vectores_reid_val, etiquetas_reid_val)
        print(
            "[INFO] Validacion Re-ID: "
            f"accuracy={metricas_validacion['reid']['accuracy']:.3f} "
            f"f1={metricas_validacion['reid']['f1_macro']:.3f}"
        )

    guardar_resumen_entrenamiento(
        configuracion,
        muestras_rostro_train,
        muestras_reid_train,
        modelos,
        muestras_rostro_validacion=muestras_rostro_val,
        muestras_reid_validacion=muestras_reid_val,
        metricas_validacion=metricas_validacion,
    )
    return modelos


def crear_dataset_demo(configuracion: Dict[str, object]) -> None:
    """Genera un dataset pequeÃ±o de prueba con colores sintÃ©ticos para validar ejecuciÃ³n."""
    rutas = configuracion.get("rutas", {})
    base_reid = Path(str(rutas.get("reidentificacionF", "datos/reidentificacionF")))
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
