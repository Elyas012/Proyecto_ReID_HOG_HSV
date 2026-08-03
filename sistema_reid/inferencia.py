"""Motor de inferencia en vivo

Flujo correcto:
1. YOLOv8n detecta la persona y genera ROI.
2. Si el rostro es visible y confiable, se usa HoG + SVM facial.
3. Si el rostro no aparece, está borroso o el SVM facial no reconoce con score suficiente,
   se activa Re-ID con HSV + SVM Re-ID.
4. Durante la ejecución, las predicciones faciales confiables alimentan el entrenamiento
   en vivo del SVM Re-ID con descriptores HSV del cuerpo completo/ropa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .caracteristicas import (
    dimension_hog_rostro,
    dimension_histograma_hsv,
    extraer_histograma_hsv,
    extraer_hog_rostro,
    parametros_hog_desde_config,
    parametros_hsv_desde_config,
    rostro_es_util,
)
from .configuracion import resolver_limites_entrenamiento
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
    caja_rostro: Optional[tuple[int, int, int, int]] = None
    detalle_rostro: str = ""
    estado_reid_vivo: Dict[str, int] = field(default_factory=dict)


class MotorInferencia:
    """Coordina detección, rostro HoG+SVM, fallback Re-ID HSV+SVM y logs."""

    def __init__(self, configuracion: Dict[str, object]) -> None:
        self.configuracion = configuracion
        rutas = configuracion.get("rutas", {})
        modelos = Path(str(rutas.get("modelos", "modelos")))

        self.modelo_rostro: Optional[ArtefactosSVM] = None
        self.modelo_reid: Optional[ArtefactosSVM] = None
        self.detector_rostros = DetectorRostros.desde_config(configuracion)
        self.parametros_hog = parametros_hog_desde_config(configuracion)
        self.parametros_hsv = parametros_hsv_desde_config(configuracion)
        self.estado_reid_vivo_actual: Dict[str, int] = {}
        self.detector_personas = DetectorPersonasYOLO(
            pesos=str(configuracion.get("yolo", {}).get("pesos", modelos / "yolov8n.pt")),
            confianza=float(configuracion.get("yolo", {}).get("confianza", 0.40)),
            tamano_imagen=int(configuracion.get("yolo", {}).get("tamano_imagen", 640)),
            dispositivo=str(configuracion.get("yolo", {}).get("dispositivo", "cpu")),
        )

        aprendizaje = configuracion.get("aprendizaje_reid_en_vivo", {})
        self.aprendizaje_reid_activo = bool(aprendizaje.get("activo", True))
        self.usar_reid_en_vivo = bool(aprendizaje.get("usar_modelo_en_vivo", True))
        self.reentrenar_combinado_al_iniciar = bool(aprendizaje.get("reentrenar_combinado_al_iniciar", True))
        self.score_rostro_min_aprendizaje = float(aprendizaje.get("score_rostro_min_aprendizaje", 0.90))
        self.margen_rostro_min_aprendizaje = float(aprendizaje.get("margen_rostro_min_aprendizaje", 0.20))
        self.entrenador_reid: Optional[EntrenadorReIDEnVivo] = None
        if self.aprendizaje_reid_activo:
            self.entrenador_reid = EntrenadorReIDEnVivo(
                carpeta_modelos=modelos,
                minimo_por_identidad=int(aprendizaje.get("minimo_por_identidad", 4)),
                reentrenar_cada=int(aprendizaje.get("reentrenar_cada", 20)),
                kernel=str(
                    configuracion.get("entrenamiento", {}).get(
                        "kernel_reid",
                        configuracion.get("entrenamiento", {}).get("kernel", "rbf"),
                    )
                ),
                probabilidad=bool(configuracion.get("entrenamiento", {}).get("probabilidad", True)),
                nombre_modelo=str(aprendizaje.get("modelo_salida", "svm_reidentificacion_en_vivo.pkl")),
                combinar_con_base=bool(aprendizaje.get("combinar_con_reid_fijo", True)),
                max_muestras_vivas_por_identidad=int(aprendizaje.get("max_muestras_vivas_por_identidad", 80)),
                parametros_hsv=self.parametros_hsv,
                dimension_descriptor=dimension_histograma_hsv(self.parametros_hsv),
                carpeta_capturas=Path(
                    str(aprendizaje.get("carpeta_capturas", rutas.get("reidentificacionF", "datos/reidentificacionF")))
                ),
                guardar_capturas=bool(aprendizaje.get("guardar_capturas", True)),
                max_imagenes_carpeta_por_identidad=int(
                    aprendizaje.get(
                        "max_imagenes_carpeta_por_identidad",
                        aprendizaje.get("max_muestras_vivas_por_identidad", 50),
                    )
                ),
                reentrenar_async=bool(aprendizaje.get("reentrenar_async", True)),
                guardar_buffer_cada=int(aprendizaje.get("guardar_buffer_cada", aprendizaje.get("reentrenar_cada", 20))),
            )

    def estado_reid_vivo(self) -> Dict[str, int]:
        """Devuelve el conteo persistente de muestras Re-ID base + vivo."""
        if self.entrenador_reid:
            self.estado_reid_vivo_actual = self.entrenador_reid.resumen()
        return dict(self.estado_reid_vivo_actual)

    def _detectar_rostro_con_zoom(self, deteccion_persona: Deteccion) -> Optional[Deteccion]:
        """Segundo intento: recorta cabeza/torso superior, amplia y busca rostro."""
        config_zoom = self.configuracion.get("rostro_en_persona", {})
        if not bool(config_zoom.get("usar_zoom_si_no_detecta", True)):
            return None

        roi_persona = deteccion_persona.roi
        if roi_persona is None or roi_persona.size == 0:
            return None

        alto, ancho = roi_persona.shape[:2]
        porcentaje_superior = float(config_zoom.get("porcentaje_superior", 0.45))
        porcentaje_superior = max(0.10, min(1.0, porcentaje_superior))
        y2_superior = max(1, min(alto, int(alto * porcentaje_superior)))
        zona_superior = roi_persona[:y2_superior, :]
        if zona_superior.size == 0:
            return None

        factor_zoom = max(1.0, min(6.0, float(config_zoom.get("factor_zoom", 3.0))))
        tamano_minimo_zoom = int(config_zoom.get("tamano_minimo_zoom", 18))
        usar_roi_zoom = bool(config_zoom.get("usar_roi_zoom_para_clasificar", True))
        ancho_zoom = max(1, int(zona_superior.shape[1] * factor_zoom))
        alto_zoom = max(1, int(zona_superior.shape[0] * factor_zoom))
        zona_zoom = cv2.resize(zona_superior, (ancho_zoom, alto_zoom), interpolation=cv2.INTER_CUBIC)

        rostros = self.detector_rostros.detectar_rostros(zona_zoom, tamano_minimo=tamano_minimo_zoom)
        if not rostros:
            return None

        rostro_zoom = rostros[0]
        zx1, zy1, zx2, zy2 = rostro_zoom.caja
        x1 = int(zx1 / factor_zoom)
        y1 = int(zy1 / factor_zoom)
        x2 = int(zx2 / factor_zoom)
        y2 = int(zy2 / factor_zoom)
        x1 = max(0, min(x1, ancho - 1))
        y1 = max(0, min(y1, alto - 1))
        x2 = max(x1 + 1, min(x2, ancho))
        y2 = max(y1 + 1, min(y2, alto))
        roi_rostro = rostro_zoom.roi if usar_roi_zoom else roi_persona[y1:y2, x1:x2]
        if roi_rostro.size == 0:
            return None
        return Deteccion(caja=(x1, y1, x2, y2), score=rostro_zoom.score, clase=0, roi=roi_rostro)

    def _caja_rostro_absoluta(self, deteccion_persona: Deteccion, deteccion_rostro: Deteccion) -> tuple[int, int, int, int]:
        """Convierte la caja de rostro relativa al ROI de persona a coordenadas del frame."""
        px1, py1, _, _ = deteccion_persona.caja
        rx1, ry1, rx2, ry2 = deteccion_rostro.caja
        return px1 + rx1, py1 + ry1, px1 + rx2, py1 + ry2

    def cargar_modelos(self) -> None:
        """Carga SVM facial, SVM Re-ID y el estado del entrenamiento Re-ID en vivo."""
        carpeta_modelos = Path(str(self.configuracion.get("rutas", {}).get("modelos", "modelos")))
        ruta_rostro = carpeta_modelos / "svm_rostro.pkl"
        ruta_reid = carpeta_modelos / "svm_reidentificacion.pkl"

        if ruta_rostro.exists():
            modelo_rostro = cargar_artefactos(ruta_rostro)
            dimension_rostro = dimension_hog_rostro(self.parametros_hog)
            if getattr(modelo_rostro.escalador, "n_features_in_", dimension_rostro) != dimension_rostro:
                print("[AVISO] modelos/svm_rostro.pkl usa un descriptor HoG anterior. Reentrena rostros para usar 128x128.")
            else:
                self.modelo_rostro = modelo_rostro
        else:
            print("[AVISO] No existe modelos/svm_rostro.pkl. Primero entrena con dataset de rostros.")

        dimension_reid = dimension_histograma_hsv(self.parametros_hsv)
        if ruta_reid.exists():
            modelo_reid_fijo = cargar_artefactos(ruta_reid)
            if getattr(modelo_reid_fijo, "tipo", "") in {"reid_hsv_svm_en_vivo", "reid_hsv_svm_combinado"}:
                print("[AVISO] modelos/svm_reidentificacion.pkl parece venir del entrenamiento en vivo anterior; no se carga como Re-ID fijo.")
            elif getattr(modelo_reid_fijo.escalador, "n_features_in_", dimension_reid) != dimension_reid:
                print("[AVISO] modelos/svm_reidentificacion.pkl usa un descriptor Re-ID anterior. Reentrena para usar hsv_espacial.")
            else:
                self.modelo_reid = modelo_reid_fijo

        if self.entrenador_reid:
            entrenamiento = self.configuracion.get("entrenamiento", {})
            limites = resolver_limites_entrenamiento(self.configuracion)
            rutas_reid = self.configuracion.get("rutas", {})
            fuentes_reid_base = [
                ("reidentificacionF", rutas_reid.get("reidentificacionF", "datos/reidentificacionF")),
            ]
            self.entrenador_reid.configurar_bases_pendientes(
                fuentes_reid_base,
                max_muestras_por_clase=limites["max_muestras_reid_por_clase"],
                min_muestras_por_clase=limites["min_muestras_reid_por_clase"],
                semilla=int(entrenamiento.get("semilla", 42)),
            )
            if self.usar_reid_en_vivo and self.reentrenar_combinado_al_iniciar:
                self.entrenador_reid.asegurar_bases_cargadas()
            self.entrenador_reid.cargar_estado()
            if self.usar_reid_en_vivo and self.reentrenar_combinado_al_iniciar:
                self.entrenador_reid.entrenar_si_es_posible()
            if self.usar_reid_en_vivo and self.entrenador_reid.modelo is not None:
                self.modelo_reid = self.entrenador_reid.modelo
            self.estado_reid_vivo_actual = self.entrenador_reid.resumen()

    def _actualizar_reid_en_vivo(
        self,
        identidad: str,
        vector_hsv: np.ndarray,
        roi_persona: np.ndarray,
        score_rostro: float,
        margen_rostro: float,
    ) -> Dict[str, int]:
        """Alimenta el SVM Re-ID con HSV cuando el rostro ya fue reconocido correctamente."""
        if not self.entrenador_reid:
            return {}
        if score_rostro < self.score_rostro_min_aprendizaje or margen_rostro < self.margen_rostro_min_aprendizaje:
            self.estado_reid_vivo_actual = self.entrenador_reid.resumen()
            return dict(self.estado_reid_vivo_actual)

        actualizado = self.entrenador_reid.agregar_muestra(identidad, vector_hsv, roi_persona)
        if (actualizado or self.entrenador_reid.modelo is not None) and self.entrenador_reid.modelo is not None:
            self.modelo_reid = self.entrenador_reid.modelo
        self.estado_reid_vivo_actual = self.entrenador_reid.resumen()
        return dict(self.estado_reid_vivo_actual)

    def _clasificar_reid(
        self,
        deteccion_persona: Deteccion,
        vector_hsv: np.ndarray,
        motivo: str,
        caja_rostro: Optional[tuple[int, int, int, int]] = None,
        detalle_rostro: str = "",
    ) -> ResultadoIdentidad:
        """Ejecuta Re-ID con HSV + SVM cuando el rostro no sirve o no reconoció."""
        score_reid_min = float(self.configuracion.get("umbrales", {}).get("score_reid", 0.65))
        estado_reid = self.estado_reid_vivo()

        if self.modelo_reid is None:
            # Comentario clave: sin SVM Re-ID entrenado no se inventa identidad; se marca desconocido.
            return ResultadoIdentidad(
                "desconocido",
                "reid_hsv_svm_sin_modelo",
                0.0,
                deteccion_persona.caja,
                {},
                detalle=f"Re-ID activado por {motivo}, pero falta SVM Re-ID entrenado",
                caja_rostro=caja_rostro,
                detalle_rostro=detalle_rostro,
                estado_reid_vivo=estado_reid,
            )

        try:
            identidad, score, ranking = predecir_con_confianza(self.modelo_reid, vector_hsv)
        except ValueError as exc:
            return ResultadoIdentidad(
                "desconocido",
                "reid_hsv_svm_reentrenar",
                0.0,
                deteccion_persona.caja,
                {},
                detalle=f"Re-ID requiere reentrenar: {exc}",
                caja_rostro=caja_rostro,
                detalle_rostro=detalle_rostro,
                estado_reid_vivo=estado_reid,
            )
        if score >= score_reid_min:
            return ResultadoIdentidad(
                identidad,
                "reid_hsv_svm",
                score,
                deteccion_persona.caja,
                ranking,
                detalle=f"Re-ID activado por {motivo}",
                caja_rostro=caja_rostro,
                detalle_rostro=detalle_rostro,
                estado_reid_vivo=estado_reid,
            )

        return ResultadoIdentidad(
            "desconocido",
            "reid_hsv_svm_score_bajo",
            score,
            deteccion_persona.caja,
            ranking,
            detalle=f"Re-ID activado por {motivo}, score bajo",
            caja_rostro=caja_rostro,
            detalle_rostro=detalle_rostro,
            estado_reid_vivo=estado_reid,
        )

    def decidir_identidad(self, deteccion_persona: Deteccion) -> ResultadoIdentidad:
        """Decide identidad respetando prioridad: rostro HoG+SVM y luego Re-ID HSV+SVM."""
        umbrales = self.configuracion.get("umbrales", {})
        score_rostro_min = float(umbrales.get("score_rostro", 0.70))
        margen_rostro_min = float(umbrales.get("margen_rostro", 0.0))
        tamano_min = int(umbrales.get("tamano_minimo_rostro", 40))
        nitidez_min = float(umbrales.get("nitidez_minima", 60.0))

        roi_reid = deteccion_persona.roi
        vector_hsv = extraer_histograma_hsv(roi_reid, **self.parametros_hsv)

        rostro = self.detector_rostros.detectar_rostro_principal(deteccion_persona.roi)
        rostro_por_zoom = False
        if rostro is None:
            rostro = self._detectar_rostro_con_zoom(deteccion_persona)
            rostro_por_zoom = rostro is not None
        if rostro is None:
            return self._clasificar_reid(deteccion_persona, vector_hsv, "rostro_no_visible_tras_zoom")

        caja_rostro = self._caja_rostro_absoluta(deteccion_persona, rostro)
        if not rostro_es_util(rostro.roi, tamano_min, nitidez_min):
            return self._clasificar_reid(
                deteccion_persona,
                vector_hsv,
                "rostro_borroso_o_pequeno",
                caja_rostro=caja_rostro,
                detalle_rostro="ROI rostro no util",
            )

        if self.modelo_rostro is None:
            return self._clasificar_reid(
                deteccion_persona,
                vector_hsv,
                "sin_svm_facial",
                caja_rostro=caja_rostro,
                detalle_rostro="ROI rostro sin SVM facial",
            )

        vector_rostro = extraer_hog_rostro(rostro.roi, **self.parametros_hog)
        identidad, score, ranking = predecir_con_confianza(self.modelo_rostro, vector_rostro)
        margen = margen_ranking(ranking)
        if score >= score_rostro_min and margen >= margen_rostro_min:
            # Comentario clave: si el rostro fue reconocido, se acepta la identidad por HoG + SVM.
            aprende_reid = score >= self.score_rostro_min_aprendizaje and margen >= self.margen_rostro_min_aprendizaje
            estado_reid = self._actualizar_reid_en_vivo(identidad, vector_hsv, roi_reid, score, margen)
            detalle = "HSV guardado para entrenar Re-ID combinado" if aprende_reid else "rostro reconocido; HSV no guardado por umbral de aprendizaje"
            if rostro_por_zoom:
                detalle = f"rostro detectado con zoom; {detalle}"
            return ResultadoIdentidad(
                identidad,
                "rostro_hog_svm_zoom" if rostro_por_zoom else "rostro_hog_svm",
                score,
                deteccion_persona.caja,
                ranking,
                detalle=detalle,
                caja_rostro=caja_rostro,
                detalle_rostro=f"ROI rostro score {score:.2f} margen {margen:.2f}",
                estado_reid_vivo=estado_reid,
            )

        # Comentario clave: rostro visible pero score bajo NO se fuerza; se pasa a Re-ID.
        return self._clasificar_reid(
            deteccion_persona,
            vector_hsv,
            f"score_o_margen_facial_bajo_{margen:.2f}",
            caja_rostro=caja_rostro,
            detalle_rostro=f"ROI rostro score {score:.2f} margen {margen:.2f}",
        )

    def procesar_frame(self, frame: np.ndarray) -> List[ResultadoIdentidad]:
        """Procesa un frame completo y devuelve identidad por cada persona detectada."""
        detecciones = self.detector_personas.detectar_personas(frame)
        resultados: List[ResultadoIdentidad] = []
        for deteccion in detecciones:
            resultados.append(self.decidir_identidad(deteccion))
        return resultados


def normalizar_modo_cajas(modo: str) -> str:
    """Normaliza alias de visualizacion de cajas."""
    modo = str(modo or "ambas").strip().lower()
    if modo in {"persona", "unificada", "unificado"}:
        return "persona"
    if modo in {"rostro", "independiente", "roi_rostro"}:
        return "rostro"
    return "ambas"


def formatear_porcentaje(score: float) -> str:
    """Muestra el score interno 0..1 como porcentaje legible."""
    return f"{max(0.0, min(1.0, float(score))) * 100:.1f}%"


def ajustar_texto_a_ancho(texto: str, ancho_maximo: int, escala: float, grosor: int) -> str:
    """Recorta texto para que no invada otras cajas o salga del frame."""
    if ancho_maximo <= 30:
        return ""
    texto = str(texto)
    if cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)[0][0] <= ancho_maximo:
        return texto

    sufijo = "..."
    disponible = max(1, ancho_maximo - cv2.getTextSize(sufijo, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)[0][0])
    acumulado = ""
    for caracter in texto:
        candidato = acumulado + caracter
        if cv2.getTextSize(candidato, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)[0][0] > disponible:
            break
        acumulado = candidato
    return (acumulado.rstrip() + sufijo) if acumulado else sufijo


def dibujar_texto_limitado(
    imagen: np.ndarray,
    texto: str,
    origen: tuple[int, int],
    ancho_maximo: int,
    escala: float,
    color: tuple[int, int, int],
    grosor: int,
) -> None:
    """Dibuja texto truncado a un ancho maximo."""
    texto_final = ajustar_texto_a_ancho(texto, ancho_maximo, escala, grosor)
    if texto_final:
        cv2.putText(imagen, texto_final, origen, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)


def dibujar_resultados(frame: np.ndarray, resultados: List[ResultadoIdentidad], modo_cajas: str = "ambas") -> np.ndarray:
    """Dibuja cajas, método usado y estado del entrenamiento Re-ID en vivo."""
    salida = frame.copy()
    alto = salida.shape[0]
    modo_cajas = normalizar_modo_cajas(modo_cajas)

    for resultado in resultados:
        x1, y1, x2, y2 = resultado.caja
        texto = f"{resultado.identidad} {formatear_porcentaje(resultado.score)} | {resultado.metodo}"

        # Comentario clave: verde para identidad aceptada, amarillo/naranja para desconocido o revisión.
        color = (0, 255, 0) if resultado.identidad != "desconocido" else (0, 180, 255)
        dibujar_persona = modo_cajas in {"persona", "ambas"} or resultado.caja_rostro is None
        dibujar_rostro = modo_cajas in {"rostro", "ambas"} and resultado.caja_rostro is not None

        if dibujar_persona:
            cv2.rectangle(salida, (x1, y1), (x2, y2), color, 2)
            ancho_texto = max(40, min(salida.shape[1] - x1 - 8, x2 - x1 + 80))
            dibujar_texto_limitado(salida, texto, (x1, max(20, y1 - 8)), ancho_texto, 0.55, color, 2)

        if dibujar_rostro and resultado.caja_rostro:
            rx1, ry1, rx2, ry2 = resultado.caja_rostro
            color_rostro = (80, 255, 120) if resultado.identidad != "desconocido" else (255, 210, 80)
            texto_rostro = texto if modo_cajas == "rostro" else (resultado.detalle_rostro or "ROI rostro")
            cv2.rectangle(salida, (rx1, ry1), (rx2, ry2), color_rostro, 2)
            ancho_texto_rostro = max(40, min(salida.shape[1] - rx1 - 8, rx2 - rx1 + 120))
            dibujar_texto_limitado(
                salida,
                texto_rostro,
                (rx1, max(20, ry1 - 6)),
                ancho_texto_rostro,
                0.45,
                color_rostro,
                1,
            )

        if resultado.detalle and dibujar_persona:
            ancho_detalle = max(40, min(salida.shape[1] - x1 - 8, x2 - x1 + 140))
            dibujar_texto_limitado(
                salida,
                resultado.detalle,
                (x1, min(alto - 10, y2 + 20)),
                ancho_detalle,
                0.45,
                color,
                1,
            )

    if resultados:
        estado = resultados[0].estado_reid_vivo
        if estado:
            texto_estado = "Re-ID vivo HSV+SVM muestras: " + ", ".join(f"{k}:{v}" for k, v in estado.items())
            dibujar_texto_limitado(salida, texto_estado, (10, alto - 15), salida.shape[1] - 20, 0.5, (255, 255, 255), 2)

    return salida
