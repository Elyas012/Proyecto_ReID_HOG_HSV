"""Motor de inferencia en vivo según la documentación oficial.

Flujo correcto:
1. YOLOv8n detecta la persona y genera ROI.
2. Si el rostro es visible y confiable, se usa HoG + SVM facial.
3. Si el rostro no aparece, está borroso o el SVM facial no reconoce con score suficiente,
   se activa Re-ID con HSV + SVM Re-ID.
4. Durante la ejecución, las predicciones faciales confiables alimentan el entrenamiento
   en vivo del SVM Re-ID con descriptores HSV del torso/ropa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, recortar_torso, rostro_es_util
from .deteccion import Deteccion, DetectorPersonasYOLO, DetectorRostros
from .modelos_svm import ArtefactosSVM, cargar_artefactos, predecir_con_confianza
from .reid_en_vivo import EntrenadorReIDEnVivo


def margen_ranking(ranking: Dict[str, float]) -> float:
    """Calcula la diferencia entre el primer y segundo candidato."""
    if len(ranking) < 2:
        return 1.0
    valores = sorted((float(valor) for valor in ranking.values()), reverse=True)
    return valores[0] - valores[1]


@dataclass
class ResultadoIdentidad:
    """Salida final para una persona detectada en un frame."""

    identidad: str
    metodo: str
    score: float
    caja: tuple[int, int, int, int]
    ranking: Dict[str, float]
    detalle: str = ""
    estado_reid_vivo: Dict[str, int] = field(default_factory=dict)


class MotorInferencia:
    """Coordina detección, rostro HoG+SVM, fallback Re-ID HSV+SVM y logs."""

    def __init__(self, configuracion: Dict[str, object]) -> None:
        self.configuracion = configuracion
        rutas = configuracion.get("rutas", {})
        modelos = Path(str(rutas.get("modelos", "modelos")))

        self.modelo_rostro: Optional[ArtefactosSVM] = None
        self.modelo_reid: Optional[ArtefactosSVM] = None
        self.detector_rostros = DetectorRostros()
        self.detector_personas = DetectorPersonasYOLO(
            pesos=str(configuracion.get("yolo", {}).get("pesos", modelos / "yolov8n.pt")),
            confianza=float(configuracion.get("yolo", {}).get("confianza", 0.40)),
            tamano_imagen=int(configuracion.get("yolo", {}).get("tamano_imagen", 640)),
            dispositivo=str(configuracion.get("yolo", {}).get("dispositivo", "cpu")),
        )

        aprendizaje = configuracion.get("aprendizaje_reid_en_vivo", {})
        self.aprendizaje_reid_activo = bool(aprendizaje.get("activo", True))
        self.usar_reid_en_vivo = bool(aprendizaje.get("usar_modelo_en_vivo", True))
        self.entrenador_reid: Optional[EntrenadorReIDEnVivo] = None
        if self.aprendizaje_reid_activo:
            self.entrenador_reid = EntrenadorReIDEnVivo(
                carpeta_modelos=modelos,
                minimo_por_identidad=int(aprendizaje.get("minimo_por_identidad", 4)),
                reentrenar_cada=int(aprendizaje.get("reentrenar_cada", 8)),
                kernel=str(configuracion.get("entrenamiento", {}).get("kernel", "rbf")),
                probabilidad=bool(configuracion.get("entrenamiento", {}).get("probabilidad", True)),
                nombre_modelo=str(aprendizaje.get("modelo_salida", "svm_reidentificacion_en_vivo.pkl")),
            )

    def cargar_modelos(self) -> None:
        """Carga SVM facial, SVM Re-ID y el estado del entrenamiento Re-ID en vivo."""
        carpeta_modelos = Path(str(self.configuracion.get("rutas", {}).get("modelos", "modelos")))
        ruta_rostro = carpeta_modelos / "svm_rostro.pkl"
        ruta_reid = carpeta_modelos / "svm_reidentificacion.pkl"

        if ruta_rostro.exists():
            self.modelo_rostro = cargar_artefactos(ruta_rostro)
        else:
            print("[AVISO] No existe modelos/svm_rostro.pkl. Primero entrena con dataset de rostros.")

        if ruta_reid.exists():
            modelo_reid_fijo = cargar_artefactos(ruta_reid)
            if getattr(modelo_reid_fijo, "tipo", "") == "reid_hsv_svm_en_vivo":
                print("[AVISO] modelos/svm_reidentificacion.pkl parece venir del entrenamiento en vivo anterior; no se carga como Re-ID fijo.")
            else:
                self.modelo_reid = modelo_reid_fijo

        if self.entrenador_reid:
            self.entrenador_reid.cargar_estado()
            if self.usar_reid_en_vivo and self.entrenador_reid.modelo is not None:
                self.modelo_reid = self.entrenador_reid.modelo

    def _actualizar_reid_en_vivo(self, identidad: str, vector_hsv: np.ndarray) -> Dict[str, int]:
        """Alimenta el SVM Re-ID con HSV cuando el rostro ya fue reconocido correctamente."""
        if not self.entrenador_reid:
            return {}

        actualizado = self.entrenador_reid.agregar_muestra(identidad, vector_hsv)
        if actualizado and self.entrenador_reid.modelo is not None:
            self.modelo_reid = self.entrenador_reid.modelo
        return self.entrenador_reid.resumen()

    def _clasificar_reid(self, deteccion_persona: Deteccion, vector_hsv: np.ndarray, motivo: str) -> ResultadoIdentidad:
        """Ejecuta Re-ID con HSV + SVM cuando el rostro no sirve o no reconoció."""
        score_reid_min = float(self.configuracion.get("umbrales", {}).get("score_reid", 0.65))

        if self.modelo_reid is None:
            # Comentario clave: sin SVM Re-ID entrenado no se inventa identidad; se marca desconocido.
            return ResultadoIdentidad(
                "desconocido",
                "reid_hsv_svm_sin_modelo",
                0.0,
                deteccion_persona.caja,
                {},
                detalle=f"Re-ID activado por {motivo}, pero falta SVM Re-ID entrenado",
            )

        identidad, score, ranking = predecir_con_confianza(self.modelo_reid, vector_hsv)
        if score >= score_reid_min:
            return ResultadoIdentidad(
                identidad,
                "reid_hsv_svm",
                score,
                deteccion_persona.caja,
                ranking,
                detalle=f"Re-ID activado por {motivo}",
            )

        return ResultadoIdentidad(
            "desconocido",
            "reid_hsv_svm_score_bajo",
            score,
            deteccion_persona.caja,
            ranking,
            detalle=f"Re-ID activado por {motivo}, score bajo",
        )

    def decidir_identidad_por_rostro(self, deteccion_rostro: Deteccion) -> ResultadoIdentidad:
        """Clasifica un ROI que ya corresponde directamente a un rostro."""
        umbrales = self.configuracion.get("umbrales", {})
        score_rostro_min = float(umbrales.get("score_rostro", 0.70))
        margen_rostro_min = float(umbrales.get("margen_rostro", 0.0))
        tamano_min = int(umbrales.get("tamano_minimo_rostro", 40))

        alto_rostro, ancho_rostro = deteccion_rostro.roi.shape[:2]
        if min(alto_rostro, ancho_rostro) < tamano_min:
            return ResultadoIdentidad(
                "desconocido",
                "rostro_directo_no_util",
                0.0,
                deteccion_rostro.caja,
                {},
                detalle="rostro directo demasiado pequeno",
            )

        if self.modelo_rostro is None:
            return ResultadoIdentidad(
                "desconocido",
                "rostro_directo_sin_modelo",
                0.0,
                deteccion_rostro.caja,
                {},
                detalle="rostro detectado, pero falta SVM facial",
            )

        vector_rostro = extraer_hog_rostro(deteccion_rostro.roi)
        identidad, score, ranking = predecir_con_confianza(self.modelo_rostro, vector_rostro)
        margen = margen_ranking(ranking)
        if score >= score_rostro_min and margen >= margen_rostro_min:
            return ResultadoIdentidad(
                identidad,
                "rostro_directo_hog_svm",
                score,
                deteccion_rostro.caja,
                ranking,
                detalle="ROI directo al rostro",
            )

        return ResultadoIdentidad(
            "desconocido",
            "rostro_directo_score_bajo",
            score,
            deteccion_rostro.caja,
            ranking,
            detalle=f"rostro detectado directo, score/margen bajo ({margen:.2f})",
        )

    def decidir_identidad(self, deteccion_persona: Deteccion) -> ResultadoIdentidad:
        """Decide identidad respetando prioridad: rostro HoG+SVM y luego Re-ID HSV+SVM."""
        umbrales = self.configuracion.get("umbrales", {})
        score_rostro_min = float(umbrales.get("score_rostro", 0.70))
        margen_rostro_min = float(umbrales.get("margen_rostro", 0.0))
        tamano_min = int(umbrales.get("tamano_minimo_rostro", 40))
        nitidez_min = float(umbrales.get("nitidez_minima", 60.0))

        torso = recortar_torso(deteccion_persona.roi)
        vector_hsv = extraer_histograma_hsv(torso)

        rostro = self.detector_rostros.detectar_rostro_principal(deteccion_persona.roi)
        if rostro is None:
            return self._clasificar_reid(deteccion_persona, vector_hsv, "rostro_no_visible")

        if not rostro_es_util(rostro.roi, tamano_min, nitidez_min):
            return self._clasificar_reid(deteccion_persona, vector_hsv, "rostro_borroso_o_pequeno")

        if self.modelo_rostro is None:
            return self._clasificar_reid(deteccion_persona, vector_hsv, "sin_svm_facial")

        vector_rostro = extraer_hog_rostro(rostro.roi)
        identidad, score, ranking = predecir_con_confianza(self.modelo_rostro, vector_rostro)
        margen = margen_ranking(ranking)
        if score >= score_rostro_min and margen >= margen_rostro_min:
            # Comentario clave: si el rostro fue reconocido, se acepta la identidad por HoG + SVM.
            estado_reid = self._actualizar_reid_en_vivo(identidad, vector_hsv)
            return ResultadoIdentidad(
                identidad,
                "rostro_hog_svm",
                score,
                deteccion_persona.caja,
                ranking,
                detalle="rostro visible reconocido; HSV guardado para entrenar Re-ID en vivo",
                estado_reid_vivo=estado_reid,
            )

        # Comentario clave: rostro visible pero score bajo NO se fuerza; se pasa a Re-ID.
        return self._clasificar_reid(deteccion_persona, vector_hsv, f"score_o_margen_facial_bajo_{margen:.2f}")

    def procesar_frame(self, frame: np.ndarray) -> List[ResultadoIdentidad]:
        """Procesa un frame completo y devuelve identidad por cada persona detectada."""
        tamano_min = int(self.configuracion.get("umbrales", {}).get("tamano_minimo_rostro", 40))
        rostros_directos = self.detector_rostros.detectar_rostros(frame, tamano_minimo=tamano_min)
        if rostros_directos:
            return [self.decidir_identidad_por_rostro(rostro) for rostro in rostros_directos]

        detecciones = self.detector_personas.detectar_personas(frame)
        resultados: List[ResultadoIdentidad] = []
        for deteccion in detecciones:
            resultados.append(self.decidir_identidad(deteccion))
        return resultados


def dibujar_resultados(frame: np.ndarray, resultados: List[ResultadoIdentidad]) -> np.ndarray:
    """Dibuja cajas, método usado y estado del entrenamiento Re-ID en vivo."""
    salida = frame.copy()
    alto = salida.shape[0]

    for resultado in resultados:
        x1, y1, x2, y2 = resultado.caja
        texto = f"{resultado.identidad} | {resultado.metodo} | {resultado.score:.2f}"

        # Comentario clave: verde para identidad aceptada, amarillo/naranja para desconocido o revisión.
        color = (0, 255, 0) if resultado.identidad != "desconocido" else (0, 180, 255)
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, 2)
        cv2.putText(salida, texto, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if resultado.detalle:
            cv2.putText(salida, resultado.detalle[:80], (x1, min(alto - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if resultados:
        estado = resultados[0].estado_reid_vivo
        if estado:
            texto_estado = "Re-ID vivo HSV+SVM muestras: " + ", ".join(f"{k}:{v}" for k, v in estado.items())
            cv2.putText(salida, texto_estado[:110], (10, alto - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return salida
