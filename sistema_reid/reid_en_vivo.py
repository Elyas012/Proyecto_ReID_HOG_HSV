"""Entrenamiento Re-ID en vivo con HSV + SVM.

Durante la ejecución, cuando el rostro sí se reconoce correctamente con HoG + SVM,
el sistema aprovecha ese momento para guardar el descriptor HSV del cuerpo completo/ropa con
la identidad confirmada. Con esas muestras se reentrena un SVM Re-ID en vivo.
Luego, si el rostro deja de verse o no supera el umbral, se usa ese SVM Re-ID.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .modelos_svm import ArtefactosSVM, cargar_artefactos, entrenar_svm, guardar_artefactos, hay_datos_para_svm
from .datos import limitar_imagenes_recientes_por_identidad


@dataclass
class EntrenadorReIDEnVivo:
    """Acumula descriptores HSV y reentrena el SVM Re-ID durante la inferencia."""

    carpeta_modelos: Path
    minimo_por_identidad: int = 4
    reentrenar_cada: int = 20
    kernel: str = "rbf"
    probabilidad: bool = True
    nombre_modelo: str = "svm_reidentificacion_en_vivo.pkl"
    combinar_con_base: bool = True
    max_muestras_vivas_por_identidad: int = 80
    parametros_hsv: Dict[str, object] = field(default_factory=dict)
    dimension_descriptor: Optional[int] = None
    carpeta_capturas: Optional[Path] = None
    guardar_capturas: bool = True
    max_imagenes_carpeta_por_identidad: int = 50
    reentrenar_async: bool = True
    guardar_buffer_cada: int = 4
    vectores_base: List[np.ndarray] = field(default_factory=list)
    etiquetas_base: List[str] = field(default_factory=list)
    vectores: List[np.ndarray] = field(default_factory=list)
    etiquetas: List[str] = field(default_factory=list)
    muestras_nuevas: int = 0
    modelo: Optional[ArtefactosSVM] = None
    _reentrenando: bool = False
    _lock_reentrenamiento: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def ruta_buffer(self) -> Path:
        """Ruta donde se guardan las muestras HSV recolectadas durante ejecución."""
        return self.carpeta_modelos / "buffer_reid_hsv_en_vivo.npz"

    @property
    def ruta_modelo(self) -> Path:
        """Ruta del SVM Re-ID que se actualiza durante la inferencia."""
        return self.carpeta_modelos / self.nombre_modelo

    def cargar_base_reid(
        self,
        carpeta_reid: str | Path,
        max_muestras_por_clase: int = 0,
        min_muestras_por_clase: int = 0,
        semilla: int = 42,
    ) -> None:
        """Carga una carpeta Re-ID base para combinarla con el buffer vivo."""
        self.cargar_bases_reid(
            [("reid_base", carpeta_reid)],
            max_muestras_por_clase=max_muestras_por_clase,
            min_muestras_por_clase=min_muestras_por_clase,
            semilla=semilla,
        )

    def cargar_bases_reid(
        self,
        carpetas_reid: Sequence[Tuple[str, str | Path]],
        max_muestras_por_clase: int = 0,
        min_muestras_por_clase: int = 0,
        semilla: int = 42,
    ) -> None:
        """Carga varias carpetas Re-ID base para combinarlas con el buffer vivo."""
        self.vectores_base = []
        self.etiquetas_base = []
        if not self.combinar_con_base:
            return

        try:
            from .datos import construir_dataset_reid
        except Exception as exc:
            print(f"[AVISO] No se pudo importar cargador Re-ID para combinar: {exc}")
            return

        for nombre_fuente, carpeta_reid in carpetas_reid:
            try:
                vectores, etiquetas, _ = construir_dataset_reid(
                    carpeta_reid,
                    max_muestras_por_clase=max_muestras_por_clase,
                    min_muestras_por_clase=0,
                    semilla=semilla,
                    parametros_hsv=self.parametros_hsv,
                    tipo=str(nombre_fuente),
                )
            except Exception as exc:
                print(f"[AVISO] No se pudo cargar Re-ID {nombre_fuente} para combinar: {exc}")
                continue

            if len(vectores) == 0:
                continue
            self.vectores_base.extend(np.asarray(vector, dtype="float32").ravel() for vector in vectores)
            self.etiquetas_base.extend(str(etiqueta) for etiqueta in etiquetas)
            print(f"[INFO] Re-ID {nombre_fuente} cargado para combinar: {self._contar_etiquetas(etiquetas)}")

        if min_muestras_por_clase > 0 and self.etiquetas_base:
            conteo = self._contar_etiquetas(self.etiquetas_base)
            pares = [
                (vector, etiqueta)
                for vector, etiqueta in zip(self.vectores_base, self.etiquetas_base)
                if conteo.get(etiqueta, 0) >= min_muestras_por_clase
            ]
            omitidas = {etiqueta: total for etiqueta, total in conteo.items() if total < min_muestras_por_clase}
            if omitidas:
                print(f"[AVISO] Re-ID combinado omitio identidades base por minimo de muestras: {omitidas}")
            self.vectores_base = [vector for vector, _ in pares]
            self.etiquetas_base = [etiqueta for _, etiqueta in pares]

        if self.etiquetas_base:
            print(f"[INFO] Re-ID base total cargado para combinar: {self.resumen_base()}")

    def _contar_etiquetas(self, etiquetas: Sequence[str]) -> Dict[str, int]:
        """Cuenta etiquetas con orden estable para diagnostico."""
        conteo: Dict[str, int] = {}
        for etiqueta in etiquetas:
            etiqueta = str(etiqueta)
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        return {etiqueta: conteo[etiqueta] for etiqueta in sorted(conteo)}

    def cargar_estado(self) -> None:
        """Carga buffer HSV y SVM Re-ID previamente guardados, si existen."""
        self.carpeta_modelos.mkdir(parents=True, exist_ok=True)
        dimension_base = len(self.vectores_base[0]) if self.vectores_base else self.dimension_descriptor

        if self.ruta_buffer.exists():
            datos = np.load(self.ruta_buffer, allow_pickle=True)
            self.vectores = [v.astype("float32") for v in datos["vectores"]]
            self.etiquetas = [str(e) for e in datos["etiquetas"]]
            if dimension_base is not None:
                pares = [
                    (vector, etiqueta)
                    for vector, etiqueta in zip(self.vectores, self.etiquetas)
                    if vector.size == dimension_base
                ]
                omitidas = len(self.vectores) - len(pares)
                self.vectores = [vector for vector, _ in pares]
                self.etiquetas = [etiqueta for _, etiqueta in pares]
                if omitidas:
                    print(f"[AVISO] Buffer Re-ID vivo omitio {omitidas} vectores con descriptor anterior. Reentrena para regenerarlos.")

        if self.ruta_modelo.exists():
            objeto = cargar_artefactos(self.ruta_modelo)
            if isinstance(objeto, ArtefactosSVM):
                esperadas = getattr(objeto.escalador, "n_features_in_", None)
                if dimension_base is not None and esperadas is not None and int(esperadas) != dimension_base:
                    print("[AVISO] Modelo Re-ID vivo anterior incompatible con el descriptor actual. Se ignorara hasta reentrenar.")
                    return
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

    def _limitar_buffer_vivo(self) -> None:
        """Conserva solo las muestras vivas mas recientes por identidad."""
        maximo = int(self.max_muestras_vivas_por_identidad)
        if maximo <= 0:
            return

        conteo: Dict[str, int] = {}
        pares_reversos = []
        for vector, etiqueta in reversed(list(zip(self.vectores, self.etiquetas))):
            if conteo.get(etiqueta, 0) >= maximo:
                continue
            pares_reversos.append((vector, etiqueta))
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1

        pares = list(reversed(pares_reversos))
        self.vectores = [vector for vector, _ in pares]
        self.etiquetas = [etiqueta for _, etiqueta in pares]

    def _dataset_entrenamiento(self) -> tuple[np.ndarray, np.ndarray]:
        """Une dataset fijo y buffer vivo para entrenar un solo SVM Re-ID combinado."""
        if self.combinar_con_base:
            vectores = [*self.vectores_base, *self.vectores]
            etiquetas = [*self.etiquetas_base, *self.etiquetas]
        else:
            vectores = list(self.vectores)
            etiquetas = list(self.etiquetas)
        return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas)

    def _nombre_seguro(self, texto: str) -> str:
        """Limpia nombres para carpetas/archivos en Windows."""
        limpio = re.sub(r'[<>:"/\\|?*]+', "_", str(texto)).strip()
        limpio = re.sub(r"\s+", " ", limpio)
        return limpio or "desconocido"

    def guardar_captura_reid(self, identidad: str, imagen_bgr: Optional[np.ndarray]) -> Optional[Path]:
        """Guarda la imagen de cuerpo completo capturada para reentrenamiento posterior."""
        if not self.guardar_capturas or imagen_bgr is None or imagen_bgr.size == 0 or self.carpeta_capturas is None:
            return None

        identidad_segura = self._nombre_seguro(identidad)
        carpeta_identidad = self.carpeta_capturas / identidad_segura
        carpeta_identidad.mkdir(parents=True, exist_ok=True)
        marca_tiempo = time.strftime("%Y%m%d_%H%M%S")
        milisegundos = int((time.time() % 1) * 1000)
        nombre_archivo = f"{identidad_segura}_reid_vivo_{marca_tiempo}_{milisegundos:03d}_{len(self.vectores):05d}.jpg"
        ruta = carpeta_identidad / nombre_archivo

        ok, codificada = cv2.imencode(".jpg", imagen_bgr)
        if not ok:
            return None
        codificada.tofile(str(ruta))
        limitar_imagenes_recientes_por_identidad(carpeta_identidad, int(self.max_imagenes_carpeta_por_identidad))
        return ruta

    def agregar_muestra(self, identidad: str, vector_hsv: np.ndarray, imagen_bgr: Optional[np.ndarray] = None) -> bool:
        """Agrega una muestra HSV de cuerpo completo asociada a una identidad ya reconocida por rostro."""
        vector = np.asarray(vector_hsv, dtype="float32").ravel()
        self.vectores.append(vector)
        self.etiquetas.append(str(identidad))
        self.guardar_captura_reid(identidad, imagen_bgr)
        self._limitar_buffer_vivo()
        self.muestras_nuevas += 1
        guardar_cada = max(1, int(self.guardar_buffer_cada))
        debe_reentrenar = self.muestras_nuevas >= self.reentrenar_cada
        if debe_reentrenar or self.muestras_nuevas % guardar_cada == 0:
            self.guardar_buffer()

        # Comentario clave: se reentrena por lotes para no frenar cada frame de la cámara.
        if debe_reentrenar and self.reentrenar_async:
            return self.entrenar_en_segundo_plano()
        if debe_reentrenar:
            return self.entrenar_si_es_posible()
        return False

    def entrenar_en_segundo_plano(self) -> bool:
        """Lanza el reentrenamiento Re-ID sin bloquear el video."""
        with self._lock_reentrenamiento:
            if self._reentrenando:
                return False
            self._reentrenando = True

        def tarea() -> None:
            try:
                self.entrenar_si_es_posible()
            except Exception as exc:
                print(f"[AVISO] Reentrenamiento Re-ID en segundo plano fallo: {exc}")
            finally:
                with self._lock_reentrenamiento:
                    self._reentrenando = False

        threading.Thread(target=tarea, name="reid-vivo-entrenamiento", daemon=True).start()
        return False

    def entrenar_si_es_posible(self) -> bool:
        """Entrena o actualiza el SVM Re-ID si ya hay datos suficientes."""
        vectores, etiquetas = self._dataset_entrenamiento()
        valido, razon = hay_datos_para_svm(etiquetas, self.minimo_por_identidad)
        if not valido:
            return False

        self.modelo = entrenar_svm(
            vectores,
            etiquetas,
            tipo="reid_hsv_svm_combinado" if self.combinar_con_base else "reid_hsv_svm_en_vivo",
            kernel=self.kernel,
            probabilidad=self.probabilidad,
        )
        guardar_artefactos(self.modelo, self.ruta_modelo)
        self.muestras_nuevas = 0
        print(
            "[OK] SVM Re-ID actualizado "
            f"con {len(etiquetas)} muestras HSV ({len(self.etiquetas_base)} base + {len(self.etiquetas)} vivo)."
        )
        return True

    def resumen_base(self) -> Dict[str, int]:
        """Devuelve conteo de muestras HSV del dataset fijo."""
        conteo: Dict[str, int] = {}
        for etiqueta in self.etiquetas_base:
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        return conteo

    def resumen(self) -> Dict[str, int]:
        """Devuelve conteo total de muestras HSV por identidad."""
        conteo: Dict[str, int] = {}
        for etiqueta in [*self.etiquetas_base, *self.etiquetas]:
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
        return conteo
