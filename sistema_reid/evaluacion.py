"""Métricas y reportes para evaluar el sistema Re-ID."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .modelos_svm import ArtefactosSVM, predecir_con_confianza


def evaluar_predicciones(y_real: Sequence[str], y_predicho: Sequence[str]) -> Dict[str, object]:
    """Calcula accuracy, F1 macro y reporte por clase."""
    if len(y_real) == 0:
        raise ValueError("No hay etiquetas reales para evaluar.")

    # Comentario clave: F1 macro evita que una identidad con muchas imágenes tape errores en otras clases.
    return {
        "accuracy": float(accuracy_score(y_real, y_predicho)),
        "f1_macro": float(f1_score(y_real, y_predicho, average="macro", zero_division=0)),
        "reporte": classification_report(y_real, y_predicho, zero_division=0),
        "matriz_confusion": confusion_matrix(y_real, y_predicho).tolist(),
    }


def evaluar_modelo(artefactos: ArtefactosSVM, vectores: np.ndarray, etiquetas: Sequence[str]) -> Dict[str, object]:
    """Evalúa un SVM usando vectores de prueba."""
    predicciones = []
    for vector in vectores:
        identidad, _, _ = predecir_con_confianza(artefactos, vector)
        # Comentario clave: se evalúa la clase final, no el score, porque accuracy necesita etiquetas.
        predicciones.append(identidad)
    return evaluar_predicciones(etiquetas, predicciones)


def guardar_reporte_texto(metricas: Dict[str, object], ruta_salida: str | Path) -> None:
    """Guarda las métricas principales en un archivo de texto."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        archivo.write(f"Accuracy: {metricas.get('accuracy', 0):.4f}\n")
        archivo.write(f"F1 macro: {metricas.get('f1_macro', 0):.4f}\n\n")
        archivo.write(str(metricas.get("reporte", "")))
        archivo.write("\nMatriz de confusión:\n")
        archivo.write(str(metricas.get("matriz_confusion", [])))

    # Comentario clave: el reporte queda listo para anexarlo al informe de laboratorio.
