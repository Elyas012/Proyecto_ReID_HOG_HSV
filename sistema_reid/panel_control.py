"""Panel lateral de diagnostico y controles para la inferencia en vivo."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from .inferencia import ResultadoIdentidad, formatear_porcentaje


class PanelControlInferencia:
    """Dibuja un panel lateral y actualiza umbrales con trackbars de OpenCV."""

    def __init__(
        self,
        nombre_ventana: str,
        configuracion: Dict[str, object],
        ancho: int = 500,
        ventana_autosize: bool = False,
    ) -> None:
        self.nombre_ventana = nombre_ventana
        self.configuracion = configuracion
        self.ancho = ancho
        self.ventana_autosize = ventana_autosize
        self.diagnostico = self._cargar_diagnostico()
        self.resumen_entrenamiento = self._cargar_resumen_entrenamiento()
        self.sliders_activos = False
        self.scroll_y = 0
        self.max_scroll_y = 0
        self._preparar_ventana()

    def _preparar_ventana(self) -> None:
        try:
            modo_ventana = cv2.WINDOW_AUTOSIZE if self.ventana_autosize else cv2.WINDOW_NORMAL
            cv2.namedWindow(self.nombre_ventana, modo_ventana)
            self._crear_sliders()
            cv2.setMouseCallback(self.nombre_ventana, self._manejar_mouse)
            self.sliders_activos = True
        except cv2.error as exc:
            print(f"[AVISO] Panel sin sliders de OpenCV: {exc}")
            self.sliders_activos = False

    def _manejar_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Permite desplazar el contenido del panel con la rueda del mouse."""
        if event != getattr(cv2, "EVENT_MOUSEWHEEL", -1) or self.max_scroll_y <= 0:
            return
        try:
            delta = cv2.getMouseWheelDelta(flags)
        except Exception:
            delta = flags
        direccion = -1 if delta > 0 else 1
        self.scroll_y = max(0, min(self.max_scroll_y, self.scroll_y + direccion * 60))

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
        self.resumen_entrenamiento = self._cargar_resumen_entrenamiento()

    def _cargar_diagnostico(self) -> Optional[Dict[str, object]]:
        ruta = Path(str(self.configuracion.get("rutas", {}).get("reportes", "reportes"))) / "diagnostico_rostro.json"
        if not ruta.exists():
            return None
        try:
            with ruta.open("r", encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return None

    def _cargar_resumen_entrenamiento(self) -> Optional[Dict[str, object]]:
        rutas = self.configuracion.get("rutas", {})
        ruta = Path(str(rutas.get("reportes", "reportes"))) / "resumen_entrenamiento.json"
        if not ruta.exists():
            return self._cargar_resumen_desde_metadata()
        try:
            with ruta.open("r", encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return self._cargar_resumen_desde_metadata()

    def _cargar_resumen_desde_metadata(self) -> Optional[Dict[str, object]]:
        ruta = Path(str(self.configuracion.get("rutas", {}).get("registros", "registros"))) / "metadata_entrenamiento.csv"
        if not ruta.exists():
            return None
        conteo_rostro: Dict[str, int] = {}
        conteo_reid: Dict[str, int] = {}
        try:
            with ruta.open("r", encoding="utf-8", newline="") as archivo:
                for fila in csv.DictReader(archivo):
                    identidad = str(fila.get("identidad", "")).strip()
                    tipo = str(fila.get("tipo", "")).strip()
                    if not identidad:
                        continue
                    if tipo == "rostro":
                        conteo_rostro[identidad] = conteo_rostro.get(identidad, 0) + 1
                    elif tipo == "reidentificacion":
                        conteo_reid[identidad] = conteo_reid.get(identidad, 0) + 1
        except OSError:
            return None
        entrenamiento = self.configuracion.get("entrenamiento", {})
        return {
            "fecha": f"metadata: {ruta.name}",
            "max_muestras_por_clase": int(entrenamiento.get("max_muestras_por_clase", 0)),
            "max_muestras_rostro_por_clase": int(
                entrenamiento.get("max_muestras_rostro_por_clase", entrenamiento.get("max_muestras_por_clase", 0))
            ),
            "min_muestras_reid_por_clase": int(entrenamiento.get("min_muestras_reid_por_clase", 0)),
            "max_muestras_reid_por_clase": int(
                entrenamiento.get("max_muestras_reid_por_clase", entrenamiento.get("max_muestras_por_clase", 0))
            ),
            "kernel_rostro": str(entrenamiento.get("kernel_rostro", entrenamiento.get("kernel", "rbf"))),
            "kernel_reid": str(entrenamiento.get("kernel_reid", entrenamiento.get("kernel", "rbf"))),
            "omitir_rostros_sin_roi": bool(entrenamiento.get("omitir_rostros_sin_roi", True)),
            "conteo_rostro_usado": dict(sorted(conteo_rostro.items())),
            "conteo_reid_usado": dict(sorted(conteo_reid.items())),
            "modelos_entrenados": [],
        }

    def construir_vista(self, frame: np.ndarray, resultados: Iterable[ResultadoIdentidad], fps: float, frame_numero: int) -> np.ndarray:
        self.actualizar_configuracion()
        resultados = list(resultados)
        alto = frame.shape[0]
        alto_contenido = max(1150, alto + 1)
        panel_contenido = np.full((alto_contenido, self.ancho, 3), (32, 34, 38), dtype=np.uint8)

        y = 34
        y = self._texto(panel_contenido, "Panel de control", 22, y, escala=0.72, color=(255, 255, 255), grosor=2, avance=32)
        y = self._texto(panel_contenido, f"Frame {frame_numero} | FPS {fps:.1f}", 22, y, color=(210, 210, 210), avance=30)
        estado_reid_vivo = next((resultado.estado_reid_vivo for resultado in resultados if resultado.estado_reid_vivo), {})
        if estado_reid_vivo:
            total_reid_vivo = sum(int(total) for total in estado_reid_vivo.values())
            resumen_reid = ", ".join(f"{nombre}:{total}" for nombre, total in list(estado_reid_vivo.items())[:4])
            y = self._texto(panel_contenido, f"Re-ID vivo total: {total_reid_vivo}", 22, y, color=(185, 230, 255), escala=0.48, avance=22)
            y = self._texto(panel_contenido, resumen_reid[:48], 22, y, color=(185, 230, 255), escala=0.42, avance=22)
        y = self._separador(panel_contenido, y)

        y = self._texto(panel_contenido, "Umbrales editables", 22, y, color=(255, 255, 255), grosor=2, avance=28)
        umbrales = self._umbrales()
        y = self._barra(panel_contenido, "score rostro", float(umbrales.get("score_rostro", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel_contenido, "margen rostro", float(umbrales.get("margen_rostro", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel_contenido, "score reid", float(umbrales.get("score_reid", 0.0)), 0.0, 1.0, y)
        y = self._barra(panel_contenido, "nitidez", float(umbrales.get("nitidez_minima", 0.0)), 0.0, 300.0, y)
        y = self._barra(panel_contenido, "tam rostro", float(umbrales.get("tamano_minimo_rostro", 0.0)), 0.0, 180.0, y)
        y = self._separador(panel_contenido, y)

        y = self._texto(panel_contenido, "Prediccion actual", 22, y, color=(255, 255, 255), grosor=2, avance=30)
        if resultados:
            for indice, resultado in enumerate(resultados[:3], start=1):
                y = self._texto(panel_contenido, f"{indice}. {resultado.identidad}  {formatear_porcentaje(resultado.score)}", 28, y, color=(110, 230, 150), grosor=2, avance=24)
                y = self._texto(panel_contenido, resultado.metodo, 42, y, color=(190, 190, 190), escala=0.45, avance=20)
                if resultado.ranking:
                    y = self._ranking(panel_contenido, resultado.ranking, y)
        else:
            y = self._texto(panel_contenido, "Sin detecciones", 28, y, color=(170, 170, 170), avance=28)
        y = self._separador(panel_contenido, y)

        y = self._texto(panel_contenido, "Diagnostico entrenamiento", 22, y, color=(255, 255, 255), grosor=2, avance=28)
        if self.resumen_entrenamiento:
            max_imgs = int(self.resumen_entrenamiento.get("max_muestras_por_clase", 0))
            texto_max = "todas" if max_imgs <= 0 else str(max_imgs)
            fecha = str(self.resumen_entrenamiento.get("fecha", "sin fecha"))
            omitir = bool(self.resumen_entrenamiento.get("omitir_rostros_sin_roi", True))
            validacion = float(self.resumen_entrenamiento.get("validacion", 0.0))
            por_clase = bool(self.resumen_entrenamiento.get("validacion_por_clase", True))
            metricas_val = self.resumen_entrenamiento.get("metricas_validacion", {})
            metricas_rostro = metricas_val.get("rostro", {}) if isinstance(metricas_val, dict) else {}
            y = self._texto(panel_contenido, f"Ultimo entreno: {fecha}", 28, y, color=(220, 220, 220), escala=0.45, avance=22)
            y = self._texto(panel_contenido, f"Max imgs/clase: {texto_max}", 28, y, color=(220, 220, 220), avance=24)
            y = self._texto(
                panel_contenido,
                f"Kernel rostro/reid: {self.resumen_entrenamiento.get('kernel_rostro', 'rbf')} / {self.resumen_entrenamiento.get('kernel_reid', 'rbf')}",
                28,
                y,
                color=(220, 220, 220),
                escala=0.44,
                avance=22,
            )
            max_reid = int(self.resumen_entrenamiento.get("max_muestras_reid_por_clase", 0))
            min_reid = int(self.resumen_entrenamiento.get("min_muestras_reid_por_clase", 0))
            texto_max_reid = "todas" if max_reid <= 0 else str(max_reid)
            y = self._texto(panel_contenido, f"Rango Re-ID: min {min_reid} / max {texto_max_reid}", 28, y, color=(220, 220, 220), escala=0.44, avance=22)
            y = self._texto(panel_contenido, f"Validacion: {validacion:.0%} {'por clase' if por_clase else 'global'}", 28, y, color=(220, 220, 220), escala=0.48, avance=22)
            y = self._texto(panel_contenido, f"Sin ROI: {'omitidas' if omitir else 'usadas completas'}", 28, y, color=(220, 220, 220), escala=0.48, avance=22)
            if metricas_rostro:
                y = self._texto(panel_contenido, f"Accuracy val: {float(metricas_rostro.get('accuracy', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
                y = self._texto(panel_contenido, f"F1 val: {float(metricas_rostro.get('f1_macro', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            elif self.diagnostico:
                y = self._texto(panel_contenido, f"Accuracy diag: {float(self.diagnostico.get('accuracy', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
                y = self._texto(panel_contenido, f"F1 diag: {float(self.diagnostico.get('f1_macro', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            conteo = self.resumen_entrenamiento.get("conteo_rostro_entrenamiento", self.resumen_entrenamiento.get("conteo_rostro_usado", {}))
            y = self._texto(panel_contenido, "Rostros usados entreno:", 28, y, color=(210, 210, 210), avance=22)
            for nombre, total in list(conteo.items())[:10]:
                y = self._texto(panel_contenido, f"{nombre}: {total}", 42, y, color=(185, 185, 185), escala=0.45, avance=19)
            conteo_val = self.resumen_entrenamiento.get("conteo_rostro_validacion", {})
            if conteo_val:
                y = self._texto(panel_contenido, "Rostros validacion:", 28, y, color=(210, 210, 210), avance=22)
                for nombre, total in list(conteo_val.items())[:6]:
                    y = self._texto(panel_contenido, f"{nombre}: {total}", 42, y, color=(185, 185, 185), escala=0.45, avance=19)
            conteo_reid = self.resumen_entrenamiento.get("conteo_reid_entrenamiento", self.resumen_entrenamiento.get("conteo_reid_usado", {}))
            if conteo_reid:
                y = self._texto(panel_contenido, "Re-ID usado entreno:", 28, y, color=(210, 210, 210), avance=22)
                for nombre, total in list(conteo_reid.items())[:6]:
                    y = self._texto(panel_contenido, f"{nombre}: {total}", 42, y, color=(185, 185, 185), escala=0.45, avance=19)
        elif self.diagnostico:
            max_imgs = int(self.configuracion.get("entrenamiento", {}).get("max_muestras_por_clase", 0))
            texto_max = "todas" if max_imgs <= 0 else str(max_imgs)
            y = self._texto(panel_contenido, f"Max imgs/clase: {texto_max}", 28, y, color=(220, 220, 220), avance=24)
            y = self._texto(panel_contenido, f"Accuracy rostro: {float(self.diagnostico.get('accuracy', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            y = self._texto(panel_contenido, f"F1 macro: {float(self.diagnostico.get('f1_macro', 0.0)):.3f}", 28, y, color=(220, 220, 220), avance=24)
            conteo = self.diagnostico.get("conteo_usado", {})
            y = self._texto(panel_contenido, "Muestras usadas diagnostico:", 28, y, color=(210, 210, 210), avance=22)
            for nombre, total in list(conteo.items())[:8]:
                y = self._texto(panel_contenido, f"{nombre}: {total}", 42, y, color=(185, 185, 185), escala=0.45, avance=19)
        else:
            y = self._texto(panel_contenido, "Ejecuta: --modo entrenar", 28, y, color=(170, 170, 170), avance=24)
        y = self._separador(panel_contenido, y)

        y = self._texto(panel_contenido, "Teclas: q salir | d diagnostico", 22, y, color=(180, 205, 255), escala=0.48, avance=22)
        y = self._texto(panel_contenido, "Rueda del mouse: desplazar panel", 22, y, color=(180, 205, 255), escala=0.44, avance=22)

        panel = self._recortar_panel(panel_contenido[: min(panel_contenido.shape[0], y + 28)], alto)
        return self._combinar(frame, panel)

    def construir_panel(self, resultados: Iterable[ResultadoIdentidad], fps: float, frame_numero: int, alto_visible: int = 720) -> np.ndarray:
        """Construye solo el panel para usarlo en una ventana separada."""
        alto_visible = max(240, int(alto_visible))
        frame_dummy = np.zeros((alto_visible, 1, 3), dtype=np.uint8)
        vista = self.construir_vista(frame_dummy, resultados, fps, frame_numero)
        return vista[:, 1:]

    def _recortar_panel(self, panel_contenido: np.ndarray, alto_visible: int) -> np.ndarray:
        """Recorta el panel completo a la altura visible y dibuja indicador de scroll."""
        alto_contenido = panel_contenido.shape[0]
        self.max_scroll_y = max(0, alto_contenido - alto_visible)
        self.scroll_y = max(0, min(self.scroll_y, self.max_scroll_y))

        if self.max_scroll_y <= 0:
            panel = np.full((alto_visible, self.ancho, 3), (32, 34, 38), dtype=np.uint8)
            panel[:alto_contenido] = panel_contenido
            return panel

        inicio = self.scroll_y
        fin = inicio + alto_visible
        panel = panel_contenido[inicio:fin].copy()
        if panel.shape[0] < alto_visible:
            relleno = np.full((alto_visible - panel.shape[0], self.ancho, 3), (32, 34, 38), dtype=np.uint8)
            panel = np.vstack([panel, relleno])
        self._dibujar_indicador_scroll(panel, alto_contenido, alto_visible)
        return panel

    def _dibujar_indicador_scroll(self, panel: np.ndarray, alto_contenido: int, alto_visible: int) -> None:
        """Dibuja una barra visual de desplazamiento en el borde derecho."""
        x1 = self.ancho - 12
        x2 = self.ancho - 6
        margen = 12
        alto_track = max(1, alto_visible - 2 * margen)
        alto_thumb = max(40, int(alto_track * (alto_visible / max(1, alto_contenido))))
        recorrido = max(1, alto_track - alto_thumb)
        y1 = margen + int(recorrido * (self.scroll_y / max(1, self.max_scroll_y)))
        y2 = y1 + alto_thumb
        cv2.rectangle(panel, (x1, margen), (x2, alto_visible - margen), (65, 68, 74), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (170, 190, 220), -1)

    def _combinar(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        alto = panel.shape[0]
        if alto == frame.shape[0]:
            frame_redimensionado = frame
        else:
            escala = alto / frame.shape[0]
            ancho_frame = max(1, int(frame.shape[1] * escala))
            interpolacion = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
            frame_redimensionado = cv2.resize(frame, (ancho_frame, alto), interpolation=interpolacion)
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

    def _separador(self, imagen: np.ndarray, y: int, avance: int = 26) -> int:
        cv2.line(imagen, (22, y), (self.ancho - 22, y), (75, 78, 84), 1)
        return y + avance

    def _barra(self, imagen: np.ndarray, etiqueta: str, valor: float, minimo: float, maximo: float, y: int, avance: int = 34, escala: float = 0.48) -> int:
        x = 28
        ancho = self.ancho - 170
        proporcion = 0.0 if maximo <= minimo else (valor - minimo) / (maximo - minimo)
        proporcion = max(0.0, min(1.0, proporcion))
        cv2.putText(imagen, f"{etiqueta}: {valor:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (220, 220, 220), 1, cv2.LINE_AA)
        y_barra = y + 11
        cv2.rectangle(imagen, (x, y_barra), (x + ancho, y_barra + 8), (70, 72, 78), -1)
        cv2.rectangle(imagen, (x, y_barra), (x + int(ancho * proporcion), y_barra + 8), (90, 180, 255), -1)
        return y + avance

    def _ranking(self, imagen: np.ndarray, ranking: Dict[str, float], y: int, limite: int = 4, avance: int = 18) -> int:
        mejores = sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:limite]
        for nombre, score in mejores:
            y = self._texto(imagen, f"{nombre}: {formatear_porcentaje(score)}", 58, y, color=(170, 170, 170), escala=0.40 if avance < 18 else 0.42, avance=avance)
        return y
