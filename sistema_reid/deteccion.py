"""Deteccion de personas con YOLOv8n y deteccion configurable de rostros.

YOLOv8n localiza personas; la identidad se decide despues con HoG+SVM facial
o HSV+SVM Re-ID, segun la documentacion del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .caracteristicas import Caja, recortar_caja

try:  # pragma: no cover - depende de instalacion local del usuario
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class Deteccion:
    """Representa una deteccion encontrada en un frame."""

    caja: Caja
    score: float
    clase: int
    roi: np.ndarray


class DetectorPersonasYOLO:
    """Detector de personas con YOLOv8n filtrando unicamente la clase person."""

    def __init__(self, pesos: str, confianza: float = 0.40, tamano_imagen: int = 640, dispositivo: str = "cpu") -> None:
        self.pesos = str(pesos)
        self.confianza = float(confianza)
        self.tamano_imagen = int(tamano_imagen)
        self.dispositivo = dispositivo
        self.modelo = None
        self.detector_hog = None
        self.usar_hog_respaldo = False

    def cargar_modelo(self) -> None:
        """Carga YOLOv8n desde modelos/yolov8n.pt o desde el nombre estandar yolov8n.pt."""
        if self.modelo is not None or self.usar_hog_respaldo:
            return
        if YOLO is None:
            print("[AVISO] Falta ultralytics. Se usara HOG de OpenCV como detector de respaldo.")
            self._cargar_hog_respaldo()
            return

        ruta_pesos = self.pesos if Path(self.pesos).exists() else "yolov8n.pt"
        try:
            self.modelo = YOLO(ruta_pesos)
        except Exception as exc:
            print(f"[AVISO] No se pudo cargar YOLO ({exc}). Se usara HOG de OpenCV como detector de respaldo.")
            self._cargar_hog_respaldo()

    def _cargar_hog_respaldo(self) -> None:
        """Prepara el detector HOG peatonal de OpenCV cuando YOLO no esta disponible."""
        if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            raise RuntimeError("OpenCV no incluye HOGDescriptor para usar el detector de respaldo.")
        self.detector_hog = cv2.HOGDescriptor()
        self.detector_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.usar_hog_respaldo = True

    def detectar_personas(self, frame: np.ndarray, clase_persona: int = 0) -> List[Deteccion]:
        """Detecta personas en el frame y devuelve caja, score y ROI recortado."""
        if frame is None or frame.size == 0:
            return []
        if self.modelo is None:
            self.cargar_modelo()
        if self.usar_hog_respaldo:
            return self._detectar_personas_hog(frame)
        return self._detectar_personas_yolo(frame, clase_persona)

    def _detectar_personas_yolo(self, frame: np.ndarray, clase_persona: int) -> List[Deteccion]:
        """Aplica YOLOv8n y conserva solo detecciones de personas."""
        resultados = self.modelo.predict(
            frame,
            conf=self.confianza,
            imgsz=self.tamano_imagen,
            device=self.dispositivo,
            verbose=False,
        )

        detecciones: List[Deteccion] = []
        for resultado in resultados:
            for caja_yolo in resultado.boxes:
                clase = int(caja_yolo.cls[0].item())
                if clase != clase_persona:
                    continue

                x1, y1, x2, y2 = caja_yolo.xyxy[0].cpu().numpy().astype(int).tolist()
                score = float(caja_yolo.conf[0].item())
                roi = recortar_caja(frame, (x1, y1, x2, y2))
                detecciones.append(Deteccion(caja=(x1, y1, x2, y2), score=score, clase=clase, roi=roi))
        return detecciones

    def _detectar_personas_hog(self, frame: np.ndarray) -> List[Deteccion]:
        """Detecta personas con HOG de OpenCV como respaldo offline."""
        rectangulos, pesos = self.detector_hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detecciones: List[Deteccion] = []
        for indice, (x, y, w, h) in enumerate(rectangulos):
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            score = float(pesos[indice]) if len(pesos) > indice else 1.0
            if score < self.confianza:
                continue
            roi = recortar_caja(frame, (x1, y1, x2, y2))
            detecciones.append(Deteccion(caja=(x1, y1, x2, y2), score=score, clase=0, roi=roi))
        return detecciones


class DetectorRostros:
    """Detector de rostros configurable: YuNet si esta disponible, Haar como respaldo."""

    def __init__(
        self,
        ruta_cascada: Optional[str] = None,
        tipo: str = "haar",
        ruta_yunet: Optional[str] = None,
        score_yunet: float = 0.60,
        nms_yunet: float = 0.30,
        top_k_yunet: int = 5000,
    ) -> None:
        self.tipo_solicitado = str(tipo or "haar").strip().lower()
        self.tipo_activo = "haar"
        self.detector_yunet = None
        self.score_yunet = float(score_yunet)
        self.nms_yunet = float(nms_yunet)
        self.top_k_yunet = int(top_k_yunet)

        if self.tipo_solicitado == "yunet" and self._cargar_yunet(ruta_yunet):
            self.clasificador = None
            self.tipo_activo = "yunet"
            return

        self.clasificador = self._cargar_haar(ruta_cascada)

    @classmethod
    def desde_config(cls, configuracion: dict) -> "DetectorRostros":
        """Crea el detector facial usando la seccion rostros del YAML."""
        config = configuracion.get("rostros", {}) if isinstance(configuracion, dict) else {}
        return cls(
            tipo=str(config.get("detector", "haar")),
            ruta_yunet=str(config.get("yunet_modelo", "")) or None,
            score_yunet=float(config.get("yunet_score", 0.60)),
            nms_yunet=float(config.get("yunet_nms", 0.30)),
            top_k_yunet=int(config.get("yunet_top_k", 5000)),
        )

    def _cargar_yunet(self, ruta_yunet: Optional[str]) -> bool:
        """Carga YuNet si OpenCV y el archivo ONNX estan disponibles."""
        creador = getattr(cv2, "FaceDetectorYN_create", None)
        if creador is None and hasattr(cv2, "FaceDetectorYN"):
            creador = getattr(cv2.FaceDetectorYN, "create", None)
        if creador is None:
            print("[AVISO] OpenCV no incluye FaceDetectorYN. Se usara Haar para rostros.")
            return False
        if not ruta_yunet or not Path(str(ruta_yunet)).exists():
            print(f"[AVISO] No existe modelo YuNet: {ruta_yunet}. Se usara Haar para rostros.")
            return False
        try:
            self.detector_yunet = creador(
                str(ruta_yunet),
                "",
                (320, 320),
                self.score_yunet,
                self.nms_yunet,
                self.top_k_yunet,
            )
            return True
        except Exception as exc:
            print(f"[AVISO] No se pudo cargar YuNet ({exc}). Se usara Haar para rostros.")
            self.detector_yunet = None
            return False

    def _cargar_haar(self, ruta_cascada: Optional[str]):
        """Carga Haar Cascade como detector facial de respaldo."""
        if not hasattr(cv2, "CascadeClassifier"):
            version = getattr(cv2, "__version__", "desconocida")
            raise RuntimeError(
                "La instalacion actual de OpenCV no incluye CascadeClassifier "
                f"(cv2 {version}). Reinstala OpenCV 4.x con: "
                "pip install --force-reinstall \"opencv-python>=4.8,<5\""
            )
        ruta = ruta_cascada or str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        clasificador = cv2.CascadeClassifier(ruta)
        if clasificador.empty():
            raise FileNotFoundError("No se pudo cargar el clasificador Haar de rostros.")
        return clasificador

    def detectar_rostros(self, imagen: np.ndarray, tamano_minimo: Optional[int] = None) -> List[Deteccion]:
        """Detecta rostros directamente en una imagen o frame completo."""
        if imagen is None or imagen.size == 0:
            return []
        if self.tipo_activo == "yunet":
            return self._detectar_rostros_yunet(imagen, tamano_minimo)
        return self._detectar_rostros_haar(imagen, tamano_minimo)

    def _detectar_rostros_haar(self, imagen: np.ndarray, tamano_minimo: Optional[int] = None) -> List[Deteccion]:
        """Detecta rostros con Haar Cascade."""
        alto, ancho = imagen.shape[:2]
        tamano_base = tamano_minimo or 30
        tamano = max(tamano_base, int(min(alto, ancho) * 0.08))
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(tamano, tamano))

        detecciones: List[Deteccion] = []
        for x, y, w, h in rostros:
            caja = (int(x), int(y), int(x + w), int(y + h))
            roi = recortar_caja(imagen, caja)
            detecciones.append(Deteccion(caja=caja, score=1.0, clase=0, roi=roi))
        detecciones.sort(key=lambda deteccion: deteccion.roi.shape[0] * deteccion.roi.shape[1], reverse=True)
        return detecciones

    def _detectar_rostros_yunet(self, imagen: np.ndarray, tamano_minimo: Optional[int] = None) -> List[Deteccion]:
        """Detecta rostros con YuNet de OpenCV."""
        alto, ancho = imagen.shape[:2]
        self.detector_yunet.setInputSize((ancho, alto))
        try:
            _, rostros = self.detector_yunet.detect(imagen)
        except Exception as exc:
            print(f"[AVISO] YuNet fallo durante deteccion ({exc}).")
            return []
        if rostros is None or len(rostros) == 0:
            return []

        tamano = int(tamano_minimo or 0)
        detecciones: List[Deteccion] = []
        for rostro in rostros:
            valores_caja = np.asarray(rostro[:4], dtype="float32")
            if valores_caja.size != 4 or not np.all(np.isfinite(valores_caja)):
                continue

            x, y, w, h = [float(valor) for valor in valores_caja]
            if w <= 0 or h <= 0:
                continue
            if tamano > 0 and (w < tamano or h < tamano):
                continue

            score = float(rostro[-1]) if len(rostro) >= 15 and np.isfinite(rostro[-1]) else 1.0
            if score < self.score_yunet:
                continue

            x1 = max(0, min(int(round(x)), ancho - 1))
            y1 = max(0, min(int(round(y)), alto - 1))
            x2 = max(x1 + 1, min(int(round(x + w)), ancho))
            y2 = max(y1 + 1, min(int(round(y + h)), alto))
            if x2 <= x1 or y2 <= y1:
                continue

            caja = (
                x1,
                y1,
                x2,
                y2,
            )
            roi = recortar_caja(imagen, caja)
            if roi.size == 0:
                continue
            detecciones.append(Deteccion(caja=caja, score=score, clase=0, roi=roi))
        detecciones.sort(key=lambda deteccion: deteccion.score, reverse=True)
        return detecciones

    def detectar_rostro_principal(self, roi_persona: np.ndarray) -> Optional[Deteccion]:
        """Busca el rostro mas grande dentro del ROI de una persona."""
        if roi_persona is None or roi_persona.size == 0:
            return None

        rostros = self.detectar_rostros(roi_persona, tamano_minimo=30)
        if not rostros:
            return None
        return max(rostros, key=lambda deteccion: deteccion.roi.shape[0] * deteccion.roi.shape[1])
