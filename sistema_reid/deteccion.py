"""Detección de personas con YOLOv8n y detección básica de rostros.

YOLOv8n solo localiza personas; la identidad se decide después con HoG+SVM facial
o HSV+SVM Re-ID, tal como indica la documentación del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .caracteristicas import Caja, recortar_caja

try:  # pragma: no cover - depende de instalación local del usuario
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class Deteccion:
    """Representa una detección encontrada en un frame."""

    caja: Caja
    score: float
    clase: int
    roi: np.ndarray


class DetectorPersonasYOLO:
    """Detector de personas con YOLOv8n filtrando únicamente la clase person."""

    def __init__(self, pesos: str, confianza: float = 0.40, tamano_imagen: int = 640, dispositivo: str = "cpu") -> None:
        self.pesos = str(pesos)
        self.confianza = float(confianza)
        self.tamano_imagen = int(tamano_imagen)
        self.dispositivo = dispositivo
        self.modelo = None
        self.detector_hog = None
        self.usar_hog_respaldo = False

    def cargar_modelo(self) -> None:
        """Carga YOLOv8n desde modelos/yolov8n.pt o desde el nombre estándar yolov8n.pt."""
        if self.modelo is not None or self.usar_hog_respaldo:
            return
        if YOLO is None:
            print("[AVISO] Falta ultralytics. Se usara HOG de OpenCV como detector de respaldo.")
            self._cargar_hog_respaldo()
            return

        # Comentario clave: si modelos/yolov8n.pt no existe, ultralytics puede descargar/usar yolov8n.pt.
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

                # Comentario clave: YOLO no reconoce identidad; solo entrega la ROI para HoG/HSV + SVM.
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
    """Detector liviano de rostros con Haar Cascade de OpenCV."""

    def __init__(self, ruta_cascada: Optional[str] = None) -> None:
        if not hasattr(cv2, "CascadeClassifier"):
            version = getattr(cv2, "__version__", "desconocida")
            raise RuntimeError(
                "La instalacion actual de OpenCV no incluye CascadeClassifier "
                f"(cv2 {version}). Reinstala OpenCV 4.x con: "
                "pip install --force-reinstall \"opencv-python>=4.8,<5\""
            )
        ruta = ruta_cascada or str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        self.clasificador = cv2.CascadeClassifier(ruta)
        if self.clasificador.empty():
            raise FileNotFoundError("No se pudo cargar el clasificador Haar de rostros.")

    def detectar_rostros(self, imagen: np.ndarray, tamano_minimo: Optional[int] = None) -> List[Deteccion]:
        """Detecta rostros directamente en una imagen o frame completo."""
        if imagen is None or imagen.size == 0:
            return []

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

    def detectar_rostro_principal(self, roi_persona: np.ndarray) -> Optional[Deteccion]:
        """Busca el rostro más grande dentro del ROI de una persona."""
        if roi_persona is None or roi_persona.size == 0:
            return None

        gris = cv2.cvtColor(roi_persona, cv2.COLOR_BGR2GRAY)
        rostros = self.clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(rostros) == 0:
            return None

        # Comentario clave: si hay varios rostros, se usa el de mayor área por ser el más confiable.
        x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
        caja = (int(x), int(y), int(x + w), int(y + h))
        roi = recortar_caja(roi_persona, caja)
        return Deteccion(caja=caja, score=1.0, clase=0, roi=roi)
