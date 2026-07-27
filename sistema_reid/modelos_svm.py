"""Entrenamiento, guardado y predicción con clasificadores SVM.

Regla oficial del proyecto:
- Identificación: rostro visible -> descriptor HoG -> SVM facial.
- Re-identificación: rostro no visible/no reconocido -> descriptor HSV torso/ropa -> SVM Re-ID.

No se usan perfiles como reemplazo del SVM Re-ID, porque la documentación pide que
ambos flujos terminen en clasificadores SVM.
"""

from __future__ import annotations

from collections import Counter
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


def contar_muestras_por_clase(etiquetas: Sequence[str]) -> Dict[str, int]:
    """Cuenta muestras por identidad con orden estable para reportes."""
    conteo = Counter(str(etiqueta) for etiqueta in etiquetas)
    return {clase: conteo[clase] for clase in sorted(conteo)}


def limitar_muestras_por_clase(
    vectores: np.ndarray,
    etiquetas: np.ndarray,
    maximo_por_clase: Optional[int],
    semilla: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce clases dominantes para que una identidad no arrastre el SVM."""
    if maximo_por_clase is None or maximo_por_clase <= 0:
        return vectores, etiquetas

    etiquetas = np.asarray(etiquetas)
    vectores = np.asarray(vectores, dtype="float32")
    rng = np.random.default_rng(semilla)
    indices_seleccionados = []

    for clase in sorted(set(etiquetas)):
        indices = np.flatnonzero(etiquetas == clase)
        if len(indices) > maximo_por_clase:
            indices = rng.choice(indices, size=maximo_por_clase, replace=False)
        indices_seleccionados.extend(indices.tolist())

    indices_seleccionados = np.asarray(sorted(indices_seleccionados), dtype=int)
    return vectores[indices_seleccionados], etiquetas[indices_seleccionados]


def dividir_indices_validacion_estratificada(
    etiquetas: Sequence[str],
    proporcion_validacion: float,
    semilla: int = 42,
    minimo_entrenamiento_por_clase: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separa indices train/validacion por identidad para no perder clases."""
    etiquetas = np.asarray(etiquetas)
    proporcion = max(0.0, min(0.8, float(proporcion_validacion)))
    indices_entrenamiento: list[int] = []
    indices_validacion: list[int] = []
    rng = np.random.default_rng(semilla)

    for clase in sorted(set(etiquetas)):
        indices = np.flatnonzero(etiquetas == clase)
        indices = rng.permutation(indices)
        max_validacion = max(0, len(indices) - minimo_entrenamiento_por_clase)
        total_validacion = int(round(len(indices) * proporcion))
        total_validacion = max(0, min(max_validacion, total_validacion))
        if proporcion > 0 and max_validacion > 0 and total_validacion == 0:
            total_validacion = 1

        indices_validacion.extend(indices[:total_validacion].tolist())
        indices_entrenamiento.extend(indices[total_validacion:].tolist())

    return np.asarray(sorted(indices_entrenamiento), dtype=int), np.asarray(sorted(indices_validacion), dtype=int)


def dividir_indices_validacion(
    etiquetas: Sequence[str],
    proporcion_validacion: float,
    semilla: int = 42,
    validacion_por_clase: bool = True,
    minimo_entrenamiento_por_clase: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide indices para train/validacion, por clase o global, sin vaciar clases."""
    if validacion_por_clase:
        return dividir_indices_validacion_estratificada(
            etiquetas,
            proporcion_validacion,
            semilla=semilla,
            minimo_entrenamiento_por_clase=minimo_entrenamiento_por_clase,
        )

    etiquetas = np.asarray(etiquetas)
    proporcion = max(0.0, min(0.8, float(proporcion_validacion)))
    if len(etiquetas) == 0 or proporcion <= 0:
        return np.arange(len(etiquetas), dtype=int), np.asarray([], dtype=int)

    rng = np.random.default_rng(semilla)
    objetivo_validacion = max(1, int(round(len(etiquetas) * proporcion)))
    conteo_train = Counter(str(etiqueta) for etiqueta in etiquetas)
    indices_validacion: list[int] = []

    for indice in rng.permutation(len(etiquetas)):
        if len(indices_validacion) >= objetivo_validacion:
            break
        clase = str(etiquetas[indice])
        if conteo_train[clase] <= minimo_entrenamiento_por_clase:
            continue
        indices_validacion.append(int(indice))
        conteo_train[clase] -= 1

    indices_validacion_set = set(indices_validacion)
    indices_entrenamiento = [indice for indice in range(len(etiquetas)) if indice not in indices_validacion_set]
    return np.asarray(sorted(indices_entrenamiento), dtype=int), np.asarray(sorted(indices_validacion), dtype=int)


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
    esperadas = getattr(artefactos.escalador, "n_features_in_", None)
    if esperadas is not None and vector_2d.shape[1] != int(esperadas):
        raise ValueError(
            f"El modelo {artefactos.tipo} espera {int(esperadas)} caracteristicas, "
            f"pero recibio {vector_2d.shape[1]}. Reentrena el modelo."
        )
    vector_escalado = artefactos.escalador.transform(vector_2d)
    identidad = str(artefactos.modelo.predict(vector_escalado)[0])

    if hasattr(artefactos.modelo, "predict_proba"):
        probabilidades = artefactos.modelo.predict_proba(vector_escalado)[0]
        ranking = {str(clase): float(prob) for clase, prob in zip(artefactos.modelo.classes_, probabilidades)}
        score = float(ranking.get(identidad, 0.0))
    else:
        decision = np.ravel(artefactos.modelo.decision_function(vector_escalado))
        ranking = {str(clase): float(valor) for clase, valor in zip(artefactos.modelo.classes_, decision)}
        score = float(ranking.get(identidad, np.max(decision)))

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
    kernel_rostro: Optional[str] = None,
    kernel_reid: Optional[str] = None,
    probabilidad: bool = True,
    max_muestras_por_clase: Optional[int] = None,
    max_muestras_rostro_por_clase: Optional[int] = None,
    max_muestras_reid_por_clase: Optional[int] = None,
    semilla: int = 42,
) -> Dict[str, object]:
    """Entrena SVM facial y SVM Re-ID cuando existan datos suficientes."""
    carpeta = Path(carpeta_modelos)
    carpeta.mkdir(parents=True, exist_ok=True)
    entrenados: Dict[str, object] = {}
    kernel_rostro = kernel_rostro or kernel
    kernel_reid = kernel_reid or kernel
    max_muestras_rostro_por_clase = (
        max_muestras_rostro_por_clase if max_muestras_rostro_por_clase is not None else max_muestras_por_clase
    )
    max_muestras_reid_por_clase = (
        max_muestras_reid_por_clase if max_muestras_reid_por_clase is not None else max_muestras_por_clase
    )

    if vectores_rostro is not None and etiquetas_rostro is not None and len(vectores_rostro) > 0:
        print(f"[INFO] Rostro muestras por identidad: {contar_muestras_por_clase(etiquetas_rostro)}")
        vectores_rostro, etiquetas_rostro = limitar_muestras_por_clase(
            vectores_rostro,
            etiquetas_rostro,
            max_muestras_rostro_por_clase,
            semilla,
        )
        if max_muestras_rostro_por_clase:
            print(f"[INFO] Rostro usado para entrenar: {contar_muestras_por_clase(etiquetas_rostro)}")
        valido, razon = hay_datos_para_svm(etiquetas_rostro)
        if valido:
            # Comentario clave: este modelo se usará primero cuando el rostro sea visible y confiable.
            modelo_rostro = entrenar_svm(vectores_rostro, etiquetas_rostro, "rostro_hog_svm", kernel_rostro, probabilidad)
            guardar_artefactos(modelo_rostro, carpeta / "svm_rostro.pkl")
            entrenados["svm_rostro"] = modelo_rostro
        else:
            print(f"[AVISO] SVM rostro no entrenado: {razon}.")

    if vectores_reid is not None and etiquetas_reid is not None and len(vectores_reid) > 0:
        print(f"[INFO] Re-ID muestras por identidad: {contar_muestras_por_clase(etiquetas_reid)}")
        vectores_reid, etiquetas_reid = limitar_muestras_por_clase(
            vectores_reid,
            etiquetas_reid,
            max_muestras_reid_por_clase,
            semilla,
        )
        if max_muestras_reid_por_clase:
            print(f"[INFO] Re-ID usado para entrenar: {contar_muestras_por_clase(etiquetas_reid)}")
        valido, razon = hay_datos_para_svm(etiquetas_reid)
        if valido:
            # Comentario clave: este modelo se usa solo cuando el rostro no sirve o no fue reconocido.
            modelo_reid = entrenar_svm(vectores_reid, etiquetas_reid, "reid_hsv_svm", kernel_reid, probabilidad)
            guardar_artefactos(modelo_reid, carpeta / "svm_reidentificacion.pkl")
            entrenados["svm_reidentificacion"] = modelo_reid
        else:
            print(f"[AVISO] SVM Re-ID no entrenado: {razon}.")

    return entrenados
