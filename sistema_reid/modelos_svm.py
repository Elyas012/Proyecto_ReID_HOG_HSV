"""Entrenamiento, guardado y predicción con clasificadores SVM.

Regla oficial del proyecto:
- Identificación: rostro visible -> descriptor HoG -> SVM facial.
- Re-identificación: rostro no visible/no reconocido -> descriptor HSV torso/ropa -> SVM Re-ID.

No se usan perfiles como reemplazo del SVM Re-ID, porque la documentación pide que
ambos flujos terminen en clasificadores SVM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ArtefactosSVM:
    """Agrupa un modelo SVM, su escalador y los nombres de clases."""

    modelo: SVC
    escalador: StandardScaler
    clases: np.ndarray
    tipo: str


def hay_datos_para_svm(etiquetas: Sequence[str], minimo_por_clase: int = 2) -> Tuple[bool, str]:
    """Valida si existen suficientes etiquetas para entrenar un SVM multiclase."""
    etiquetas = list(etiquetas)
    if not etiquetas:
        return False, "no hay muestras"

    clases = sorted(set(etiquetas))
    if len(clases) < 2:
        # Comentario clave: un SVM de identidad necesita comparar al menos dos personas.
        return False, "SVM necesita al menos dos identidades registradas"

    faltantes = [clase for clase in clases if etiquetas.count(clase) < minimo_por_clase]
    if faltantes:
        return False, f"faltan mínimo {minimo_por_clase} muestras en: {', '.join(faltantes)}"

    return True, "ok"


def entrenar_svm(
    vectores: np.ndarray,
    etiquetas: np.ndarray,
    tipo: str,
    kernel: str = "rbf",
    probabilidad: bool = True,
) -> ArtefactosSVM:
    """Entrena un SVM multiclase con vectores HoG o HSV."""
    vectores = np.asarray(vectores, dtype="float32")
    etiquetas = np.asarray(etiquetas)

    valido, razon = hay_datos_para_svm(etiquetas)
    if not valido:
        raise ValueError(f"No se puede entrenar SVM {tipo}: {razon}.")

    escalador = StandardScaler()
    vectores_escalados = escalador.fit_transform(vectores)

    # Comentario clave: probability=True permite obtener un score para aplicar umbrales.
    modelo = SVC(kernel=kernel, probability=probabilidad, class_weight="balanced")
    modelo.fit(vectores_escalados, etiquetas)
    return ArtefactosSVM(modelo=modelo, escalador=escalador, clases=modelo.classes_, tipo=tipo)


def predecir_con_confianza(artefactos: ArtefactosSVM, vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """Predice identidad, score y ranking usando el SVM entrenado."""
    vector_2d = np.asarray(vector, dtype="float32").reshape(1, -1)
    vector_escalado = artefactos.escalador.transform(vector_2d)

    if hasattr(artefactos.modelo, "predict_proba"):
        probabilidades = artefactos.modelo.predict_proba(vector_escalado)[0]
        indice = int(np.argmax(probabilidades))
        score = float(probabilidades[indice])
        ranking = {str(clase): float(prob) for clase, prob in zip(artefactos.modelo.classes_, probabilidades)}
    else:
        decision = np.ravel(artefactos.modelo.decision_function(vector_escalado))
        indice = int(np.argmax(decision))
        score = float(decision[indice])
        ranking = {str(clase): float(valor) for clase, valor in zip(artefactos.modelo.classes_, decision)}

    identidad = str(artefactos.modelo.classes_[indice])
    return identidad, score, ranking


def guardar_artefactos(artefactos: object, ruta_salida: str | Path) -> None:
    """Guarda modelos SVM, escaladores y clases en un archivo .pkl."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Comentario clave: se guarda el SVM junto al scaler para usar la misma normalización en inferencia.
    joblib.dump(artefactos, ruta)


def cargar_artefactos(ruta_modelo: str | Path) -> object:
    """Carga un archivo .pkl generado por el entrenamiento."""
    ruta = Path(ruta_modelo)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo: {ruta}")
    return joblib.load(ruta)


def entrenar_modelos_principales(
    vectores_rostro: Optional[np.ndarray],
    etiquetas_rostro: Optional[np.ndarray],
    vectores_reid: Optional[np.ndarray],
    etiquetas_reid: Optional[np.ndarray],
    carpeta_modelos: str | Path,
    kernel: str = "rbf",
    probabilidad: bool = True,
) -> Dict[str, object]:
    """Entrena SVM facial y SVM Re-ID cuando existan datos suficientes."""
    carpeta = Path(carpeta_modelos)
    carpeta.mkdir(parents=True, exist_ok=True)
    entrenados: Dict[str, object] = {}

    if vectores_rostro is not None and etiquetas_rostro is not None and len(vectores_rostro) > 0:
        valido, razon = hay_datos_para_svm(etiquetas_rostro)
        if valido:
            # Comentario clave: este modelo se usará primero cuando el rostro sea visible y confiable.
            modelo_rostro = entrenar_svm(vectores_rostro, etiquetas_rostro, "rostro_hog_svm", kernel, probabilidad)
            guardar_artefactos(modelo_rostro, carpeta / "svm_rostro.pkl")
            entrenados["svm_rostro"] = modelo_rostro
        else:
            print(f"[AVISO] SVM rostro no entrenado: {razon}.")

    if vectores_reid is not None and etiquetas_reid is not None and len(vectores_reid) > 0:
        valido, razon = hay_datos_para_svm(etiquetas_reid)
        if valido:
            # Comentario clave: este modelo se usa solo cuando el rostro no sirve o no fue reconocido.
            modelo_reid = entrenar_svm(vectores_reid, etiquetas_reid, "reid_hsv_svm", kernel, probabilidad)
            guardar_artefactos(modelo_reid, carpeta / "svm_reidentificacion.pkl")
            entrenados["svm_reidentificacion"] = modelo_reid
        else:
            print(f"[AVISO] SVM Re-ID no entrenado: {razon}.")

    return entrenados
