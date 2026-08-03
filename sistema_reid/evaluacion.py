"""Metricas y reportes para evaluar el sistema Re-ID."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .caracteristicas import parametros_hog_desde_config
from .datos import construir_dataset_rostros, listar_imagenes_por_identidad
from .modelos_svm import ArtefactosSVM, cargar_artefactos, dividir_indices_validacion, predecir_con_confianza


def evaluar_predicciones(
    y_real: Sequence[str],
    y_predicho: Sequence[str],
    clases: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Calcula accuracy, F1 macro, reporte por clase y matriz de confusion."""
    if len(y_real) == 0:
        raise ValueError("No hay etiquetas reales para evaluar.")

    etiquetas = [str(clase) for clase in clases] if clases is not None else sorted(set([*y_real, *y_predicho]))
    return {
        "accuracy": float(accuracy_score(y_real, y_predicho)),
        "f1_macro": float(f1_score(y_real, y_predicho, average="macro", zero_division=0)),
        "clases": etiquetas,
        "reporte": classification_report(y_real, y_predicho, labels=etiquetas, zero_division=0),
        "matriz_confusion": confusion_matrix(y_real, y_predicho, labels=etiquetas).tolist(),
    }


def evaluar_modelo(artefactos: ArtefactosSVM, vectores: np.ndarray, etiquetas: Sequence[str]) -> Dict[str, object]:
    """Evalua un SVM usando vectores de prueba."""
    predicciones = []
    for vector in vectores:
        identidad, _, _ = predecir_con_confianza(artefactos, vector)
        predicciones.append(identidad)
    return evaluar_predicciones(etiquetas, predicciones, getattr(artefactos, "clases", None))


def guardar_reporte_texto(metricas: Dict[str, object], ruta_salida: str | Path) -> None:
    """Guarda las metricas principales en un archivo de texto."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        archivo.write(f"Modelo: {metricas.get('modelo', '')}\n")
        archivo.write(f"Tipo: {metricas.get('tipo', '')}\n")
        archivo.write(f"Evaluacion: {metricas.get('evaluacion', '')}\n")
        archivo.write(f"Validacion: {metricas.get('validacion', 0):.2f}\n")
        archivo.write(f"Validacion por clase: {metricas.get('validacion_por_clase', True)}\n")
        archivo.write(f"Accuracy: {metricas.get('accuracy', 0):.4f}\n")
        archivo.write(f"F1 macro: {metricas.get('f1_macro', 0):.4f}\n")
        archivo.write(f"Conteo original: {metricas.get('conteo_original', {})}\n")
        archivo.write(f"Conteo usado: {metricas.get('conteo_usado', {})}\n\n")
        archivo.write(str(metricas.get("reporte", "")))
        archivo.write("\nMatriz de confusion:\n")
        archivo.write(str(metricas.get("matriz_confusion", [])))


def guardar_matriz_confusion_csv(metricas: Dict[str, object], ruta_salida: str | Path) -> None:
    """Guarda la matriz de confusion como CSV con encabezados."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    clases = [str(clase) for clase in metricas.get("clases", [])]
    matriz = metricas.get("matriz_confusion", [])

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real\\predicho", *clases])
        for clase, fila in zip(clases, matriz):
            escritor.writerow([clase, *fila])


def guardar_matriz_confusion_imagen(metricas: Dict[str, object], ruta_salida: str | Path) -> None:
    """Dibuja una matriz de confusion simple usando OpenCV."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    clases = [str(clase) for clase in metricas.get("clases", [])]
    matriz = np.asarray(metricas.get("matriz_confusion", []), dtype=np.float32)
    if matriz.size == 0:
        return

    celda = 76
    margen_x = 160
    margen_y = 110
    alto = margen_y + celda * len(clases) + 70
    ancho = margen_x + celda * len(clases) + 40
    imagen = np.full((alto, ancho, 3), 245, dtype=np.uint8)
    maximo = max(1.0, float(matriz.max()))

    cv2.putText(imagen, "Matriz de confusion - Rostro HOG + SVM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.putText(imagen, "Predicho", (margen_x + 20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(imagen, "Real", (25, margen_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)

    for indice, clase in enumerate(clases):
        x = margen_x + indice * celda
        cv2.putText(imagen, clase[:8], (x + 4, margen_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)
        cv2.putText(imagen, clase[:12], (55, margen_y + indice * celda + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)

    for fila in range(len(clases)):
        for columna in range(len(clases)):
            valor = int(matriz[fila, columna])
            intensidad = int(255 - 180 * (valor / maximo))
            color = (255, intensidad, intensidad)
            x1 = margen_x + columna * celda
            y1 = margen_y + fila * celda
            x2 = x1 + celda
            y2 = y1 + celda
            cv2.rectangle(imagen, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(imagen, str(valor), (x1 + 22, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    cv2.imwrite(str(ruta), imagen)


def _conteo_muestras(muestras: Sequence[object]) -> Dict[str, int]:
    conteo = Counter(str(muestra.identidad) for muestra in muestras)
    return {clase: conteo[clase] for clase in sorted(conteo)}


def diagnosticar_modelo_rostro(configuracion: Dict[str, object]) -> Dict[str, object]:
    """Evalua el SVM facial contra el dataset de rostro usado para entrenamiento."""
    rutas = configuracion["rutas"]
    entrenamiento = configuracion.get("entrenamiento", {})
    max_muestras = int(entrenamiento.get("max_muestras_por_clase", 0))
    semilla = int(entrenamiento.get("semilla", 42))
    proporcion_validacion = float(entrenamiento.get("validacion", 0.20))
    validacion_por_clase = bool(entrenamiento.get("validacion_por_clase", True))
    omitir_sin_roi = bool(entrenamiento.get("omitir_rostros_sin_roi", True))
    filtrar_rostros_baja_calidad = bool(entrenamiento.get("filtrar_rostros_baja_calidad", True))
    umbrales = configuracion.get("umbrales", {})
    parametros_hog = parametros_hog_desde_config(configuracion)
    ruta_modelo = Path(rutas["modelos"]) / "svm_rostro.pkl"
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe el modelo facial: {ruta_modelo}")

    muestras_originales = listar_imagenes_por_identidad(rutas["rostros"], "rostro")
    vectores, etiquetas, muestras_usadas = construir_dataset_rostros(
        rutas["rostros"],
        max_muestras_por_clase=max_muestras,
        semilla=semilla,
        omitir_sin_roi=omitir_sin_roi,
        filtrar_baja_calidad=filtrar_rostros_baja_calidad,
        tamano_minimo=int(umbrales.get("tamano_minimo_rostro", 40)),
        nitidez_minima=float(umbrales.get("nitidez_minima", 60.0)),
        parametros_hog=parametros_hog,
        configuracion=configuracion,
    )
    muestras_evaluacion = muestras_usadas
    if proporcion_validacion > 0 and len(etiquetas) > 0:
        _, indices_validacion = dividir_indices_validacion(
            etiquetas,
            proporcion_validacion,
            semilla=semilla,
            validacion_por_clase=validacion_por_clase,
        )
        if len(indices_validacion) > 0:
            vectores = vectores[indices_validacion]
            etiquetas = etiquetas[indices_validacion]
            muestras_evaluacion = [muestras_usadas[indice] for indice in indices_validacion]

    artefactos = cargar_artefactos(ruta_modelo)
    metricas = evaluar_modelo(artefactos, vectores, etiquetas)
    metricas["conteo_original"] = _conteo_muestras(muestras_originales)
    metricas["conteo_usado"] = _conteo_muestras(muestras_evaluacion)
    metricas["validacion"] = proporcion_validacion
    metricas["validacion_por_clase"] = validacion_por_clase
    metricas["evaluacion"] = "validacion" if muestras_evaluacion is not muestras_usadas else "dataset_completo"
    metricas["modelo"] = str(ruta_modelo)
    metricas["tipo"] = getattr(artefactos, "tipo", "rostro_hog_svm")

    reportes = Path(rutas.get("reportes", "reportes"))
    guardar_reporte_texto(metricas, reportes / "diagnostico_rostro.txt")
    guardar_matriz_confusion_csv(metricas, reportes / "matriz_confusion_rostro.csv")
    guardar_matriz_confusion_imagen(metricas, reportes / "matriz_confusion_rostro.jpg")
    with (reportes / "diagnostico_rostro.json").open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)
    return metricas


def resumen_dataset(configuracion: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    """Devuelve conteos rapidos de rostro y Re-ID sin extraer descriptores."""
    rutas = configuracion["rutas"]
    return {
        "rostros": _conteo_muestras(listar_imagenes_por_identidad(rutas["rostros"], "rostro")),
        "reidentificacionF": _conteo_muestras(listar_imagenes_por_identidad(rutas["reidentificacionF"], "reidentificacionF")),
    }
