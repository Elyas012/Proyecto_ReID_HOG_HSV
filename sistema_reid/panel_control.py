"""Panel lateral de diagnostico y controles para la inferencia en vivo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from .inferencia import ResultadoIdentidad


class PanelControlInferencia:
    """Dibuja un panel lateral y actualiza umbrales con trackbars de OpenCV."""

    def __init__(self, nombre_ventana: str, configuracion: Dict[str, object], ancho: int = 500) -> None:
        self.nombre_ventana = nombre_ventana
        self.configuracion = configuracion
        self.ancho = ancho
        self.diagnostico = self._cargar_diagnostico()
        self.sliders_activos = False
        self._preparar_ventana()

    def _preparar_ventana(self) -> None:
        try:
            cv2.namedWindow(self.nombre_ventana, cv2.WINDOW_NORMAL)
            self._crear_sliders()
            self.sliders_activos = True
        except cv2.error as exc:
            print(f"[AVISO] Panel sin sliders de OpenCV: {exc}")
            self.sliders_activos = False

    def _umbrales(self) -> Dict[str, object]:
        return self.configuracion.setdefault("umbrales", {})

    def _crear_sliders(self) -> None:
        umbrales = self._umbrales()
        cv2.createTrackbar("score rostro x100", self.nombre_ventana, int(float(umbrales.get("score_rostro", 0.70)) * 100), 100, lambda _: None)
        cv2.createTrackbar("margen rostro x100", self.nombre_ventana, int(float(umbrales.get("margen_rostro", 0.12)) * 100), 100, lambda _: None)
        cv2.createTrackbar("score reid x100", self.nombre_ventana, int(float(umbrales.get("score_reid", 0.65)) * 100), 100, lambda _: None)
        cv2.createTrackbar("nitidez rostro", self.nombre_ventana, int(float(umbrales.get("nitidez_minima", 60.0))), 300, lambda _: None)
        cv2.createTrackbar("tam rostro", self.nombre_ventana, int(float(umbrales.get("tamano_minimo_rostro", 40))), 180, lambda _: None)

    def actualizar_configuracion(self) -> None:
        if not self.sliders_activos:
            return

        umbrales = self._umbrales()
        try:
            umbrales["score_rostro"] = cv2.getTrackbarPos("score rostro x100", self.nombre_ventana) / 100.0
            umbrales["margen_rostro"] = cv2.getTrackbarPos("margen rostro x100", self.nombre_ventana) / 100.0
            umbrales["score_reid"] = cv2.getTrackbarPos("score reid x100", self.nombre_ventana) / 100.0
            umbrales["nitidez_minima"] = float(cv2.getTrackbarPos("nitidez rostro", self.nombre_ventana))
            umbrales["tamano_minimo_rostro"] = max(10, int(cv2.getTrackbarPos("tam rostro", self.nombre_ventana)))
        except cv2.error:
            self.sliders_activos = False

    def recargar_diagnostico(self) -> None:
        self.diagnostico = self._cargar_diagnostico()

    def _cargar_diagnostico(self) -> Optional[Dict[str, object]]:
        ruta = Path(str(self.configuracion.get("rutas", {}).get("reportes", "reportes"))) / "diagnostico_rostro.json"
        if not ruta.exists():
            return None
        try:
            with ruta.open("r", encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return None

    def construir_vista(self, frame: np.ndarray, resultados: Iterable[ResultadoIdentidad], fps: float, frame_numero: int) -> np.ndarray:
        self.actualizar_configuracion()
        resultados = list(resultados)
        alto = max(520, frame.shape[0])
        panel = np.full((alto, self.ancho, 3), (32, 34, 38), dtype=np.uint8)

        y = 34
        y = self._texto(panel, "Panel de control", 22, y, escala=0.72, color=(255, 255, 255), grosor=2, avance=32)
        y = self._texto(panel, f"Frame {frame_numero} | FPS {fps:.1f}", 22, y, color=(210, 210, 210), avance=30)
        y = self._separador(panel, y)

        y = self._texto(panel, "Umbrales editables", 22, y, color=(255, 255, 255), grosor=2, avance=28)
        umbrales = self._umbrales()
        y = self._barra(panel, "score rostro", float(umbrales.get("score_rostro", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel, "margen rostro", float(umbrales.get("margen_rostro", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel, "score reid", float(umbrales.get("score_reid", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel, "nitidez", float(umbrales.get("nitidez_minima", 0.0)), 0.0, 300.0, y)
        y = self._barra(panel, "tam rostro", float(umbrales.get("tamano_minimo_rostro", 0.0)), 0.0, 180.0, y)
        y = self._separador(panel, y)

        y = self._texto(panel, "Prediccion actual", 22, y, color=(255, 255, 255), grosor=2, avance=30)
        if resultados:
            for indice, resultado in enumerate(resultados[:3], start=1):
                y = self._texto(panel, f"{indice}. {resultado.identidad}  {resultado.score:.3f}", 28, y, color=(110, 230, 150), grosor=2, avance=24)
                y = self._texto(panel, resultado.metodo, 42, y, color=(190, 190, 190), escala=0.45, avance=20)
                if resultado.ranking:
                    y = self._ranking(panel, resultado.ranking, y)
        else:
            y = self._texto(panel, "Sin detecciones", 28, y, color=(170, 170, 170), avance=28)
        y = self._separador(panel, y)

        y = self._texto(panel, "Diagnostico entrenamiento", 22, y, color=(255, 255, 255), grosor=2, avance=28)
        if self.diagnostico:
            max_imgs = int(self.configuracion.get("entrenamiento", {}).get("max_muestras_por_clase", 0))
            texto_max = "todas" if max_imgs <= 0 else str(max_imgs)
            y = self._texto(panel, f"Max imgs/clase: {texto_max}", 28, y, color=(220, 220, 220), avance=24)
            y = self._texto(panel, f"Accuracy rostro: {float(self.diagnostico.get('accuracy', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            y = self._texto(panel, f"F1 macro: {float(self.diagnostico.get('f1_macro', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            conteo = self.diagnostico.get("conteo_usado", {})
            y = self._texto(panel, "Muestras usadas:", 28, y, color=(210, 210, 210), avance=22)
            for nombre, total in list(conteo.items())[:8]:
                y = self._texto(panel, f"{nombre}: {total}", 42, y, color=(185, 185, 185), escala=0.45, avance=19)
        else:
            y = self._texto(panel, "Ejecuta: --modo diagnostico", 28, y, color=(170, 170, 170), avance=24)
        y = self._separador(panel, y)

        self._texto(panel, "Teclas: q salir | d diagnostico", 22, y, color=(180, 205, 255), escala=0.48, avance=22)
        return self._combinar(frame, panel)

    def _combinar(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        alto = panel.shape[0]
        escala = alto / frame.shape[0]
        ancho_frame = max(1, int(frame.shape[1] * escala))
        frame_redimensionado = cv2.resize(frame, (ancho_frame, alto), interpolation=cv2.INTER_AREA)
        return np.hstack([frame_redimensionado, panel])

    def _texto(
        self,
        imagen: np.ndarray,
        texto: str,
        x: int,
        y: int,
        escala: float = 0.52,
        color: tuple[int, int, int] = (235, 235, 235),
        grosor: int = 1,
        avance: int = 22,
    ) -> int:
        cv2.putText(imagen, str(texto), (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
        return y + avance

    def _separador(self, imagen: np.ndarray, y: int) -> int:
        cv2.line(imagen, (22, y), (self.ancho - 22, y), (75, 78, 84), 1)
        return y + 26

    def _barra(self, imagen: np.ndarray, etiqueta: str, valor: float, minimo: float, maximo: float, y: int) -> int:
        x = 28
        ancho = self.ancho - 170
        proporcion = 0.0 if maximo <= minimo else (valor - minimo) / (maximo - minimo)
        proporcion = max(0.0, min(1.0, proporcion))
        cv2.putText(imagen, f"{etiqueta}: {valor:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        y_barra = y + 11
        cv2.rectangle(imagen, (x, y_barra), (x + ancho, y_barra + 8), (70, 72, 78), -1)
        cv2.rectangle(imagen, (x, y_barra), (x + int(ancho * proporcion), y_barra + 8), (90, 180, 255), -1)
        return y + 34

    def _ranking(self, imagen: np.ndarray, ranking: Dict[str, float], y: int) -> int:
        mejores = sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:4]
        for nombre, score in mejores:
            y = self._texto(imagen, f"{nombre}: {score:.3f}", 58, y, color=(170, 170, 170), escala=0.42, avance=18)
        return y
