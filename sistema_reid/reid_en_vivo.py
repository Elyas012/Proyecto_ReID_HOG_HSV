"""Entrenamiento Re-ID en vivo con HSV + SVM.

Durante la ejecución, cuando el rostro sí se reconoce correctamente con HoG + SVM,
el sistema aprovecha ese momento para guardar el descriptor HSV del torso/ropa con
la identidad confirmada. Con esas muestras se reentrena un SVM Re-ID en vivo.
Luego, si el rostro deja de verse o no supera el umbral, se usa ese SVM Re-ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .modelos_svm import ArtefactosSVM, cargar_artefactos, entrenar_svm, guardar_artefactos, hay_datos_para_svm


@dataclass
class EntrenadorReIDEnVivo:
    """Acumula descriptores HSV y reentrena el SVM Re-ID durante la inferencia."""

    carpeta_modelos: Path
    minimo_por_identidad: int = 4
    reentrenar_cada: int = 8
    kernel: str = "rbf"
    probabilidad: bool = True
    vectores: List[np.ndarray] = field(default_factory=list)
    etiquetas: List[str] = field(default_factory=list)
    muestras_nuevas: int = 0
    modelo: Optional[ArtefactosSVM] = None

    @property
    def ruta_buffer(self) -> Path:
        """Ruta donde se guardan las muestras HSV recolectadas durante ejecución."""
        return self.carpeta_modelos / "buffer_reid_hsv_en_vivo.npz"

    @property
    def ruta_modelo(self) -> Path:
        """Ruta oficial del SVM Re-ID entrenado con HSV."""
        return self.carpeta_modelos / "svm_reidentificacion.pkl"

    def cargar_estado(self) -> None:
        """Carga buffer HSV y SVM Re-ID previamente guardados, si existen."""
        self.carpeta_modelos.mkdir(parents=True, exist_ok=True)

        if self.ruta_buffer.exists():
            datos = np.load(self.ruta_buffer, allow_pickle=True)
            self.vectores = [v.astype("float32") for v in datos["vectores"]]
            self.etiquetas = [str(e) for e in datos["etiquetas"]]

        if self.ruta_modelo.exists():
            objeto = cargar_artefactos(self.ruta_modelo)
            if isinstance(objeto, ArtefactosSVM):
                self.modelo = objeto

    def guardar_buffer(self) -> None:
        """Guarda las muestras HSV acumuladas para no perder el entrenamiento en vivo."""
        self.carpeta_modelos.mkdir(parents=True, exist_ok=True)
        if not self.vectores:
            return

        # Comentario clave: este buffer permite reentrenar Re-ID sin copiar imágenes manualmente.
        np.savez_compressed(
            self.ruta_buffer,
            vectores=np.asarray(self.vectores, dtype="float32"),
            etiquetas=np.asarray(self.etiquetas, dtype=object),
        )

    def agregar_muestra(self, identidad: str, vector_hsv: np.ndarray) -> bool:
        """Agrega una muestra HSV de torso/ropa asociada a una identidad ya reconocida por rostro."""
        vector = np.asarray(vector_hsv, dtype="float32").ravel()
        self.vectores.append(vector)
        self.etiquetas.append(str(identidad))
        self.muestras_nuevas += 1
        self.guardar_buffer()

        # Comentario clave: se reentrena por lotes para no frenar cada frame de la cámara.
        if self.muestras_nuevas >= self.reentrenar_cada:
            return self.entrenar_si_es_posible()
        return False

    def entrenar_si_es_posible(self) -> bool:
        """Entrena o actualiza el SVM Re-ID si ya hay datos suficientes."""
        valido, razon = hay_datos_para_svm(self.etiquetas, self.minimo_por_identidad)
        if not valido:
            return False

        vectores = np.asarray(self.vectores, dtype="float32")
        etiquetas = np.asarray(self.etiquetas)
        self.modelo = entrenar_svm(
            vectores,
            etiquetas,
            tipo="reid_hsv_svm_en_vivo",
            kernel=self.kernel,
            probabilidad=self.probabilidad,
        )
        guardar_artefactos(self.modelo, self.ruta_modelo)
        self.muestras_nuevas = 0
        print(f"[OK] SVM Re-ID actualizado en vivo con {len(self.etiquetas)} muestras HSV.")
        return True

    def resumen(self) -> Dict[str, int]:
        """Devuelve conteo de muestras HSV por identidad."""
        conteo: Dict[str, int] = {}
        for etiqueta in self.etiquetas:
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        return conteo
