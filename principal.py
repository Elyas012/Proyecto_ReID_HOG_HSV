"""Entrada principal del proyecto Re-ID PC Python + cámaras móviles.

Comandos principales:
- registrar_reid: captura cuerpo completo/ropa desde cámara para entrenar SVM Re-ID.
- registrar_rostro: captura rostros visibles para identificación facial.
- registrar: captura rostro y Re-ID en la misma sesión.
- entrenar: entrena SVM facial y SVM Re-ID con las capturas disponibles.
- inferir: procesa imagen, video, webcam o URL IP/RTSP/HTTP.
- demo: crea datos sintéticos y valida que el pipeline entrene sin cámara.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from sistema_reid.captura_tiempo_real import (
    capturar_muestras_tiempo_real,
    crear_dataset_demo,
    entrenar_desde_capturas,
)
from sistema_reid.configuracion import cargar_configuracion, crear_directorios_base
from sistema_reid.datos import cargar_imagen_bgr, guardar_imagen_bgr, listar_imagenes_por_identidad
from sistema_reid.deteccion import DetectorPersonasYOLO, DetectorRostros
from sistema_reid.evaluacion import diagnosticar_modelo_rostro
from sistema_reid.inferencia import MotorInferencia, ResultadoIdentidad, dibujar_resultados
from sistema_reid.panel_control import PanelControlInferencia
from sistema_reid.seguimiento import RastreadorSimple


def crear_parser() -> argparse.ArgumentParser:
    """Crea los argumentos de consola del sistema."""
    parser = argparse.ArgumentParser(description="Sistema Re-ID HoG/HSV/SVM para PC con cámaras móviles")
    parser.add_argument("--config", default="configuracion.yaml", help="Ruta del archivo de configuración")
    parser.add_argument(
        "--modo",
        choices=[
            "menu",
            "registrar_reid",
            "registrar_rostro",
            "registrar",
            "entrenar",
            "inferir",
            "capturar_rostros_video",
            "demo",
            "revisar",
            "diagnostico",
        ],
        default="menu",
        help="Acción principal a ejecutar",
    )
    parser.add_argument("--fuente", default=None, help="Imagen, video, URL IP/RTSP/HTTP o índice de cámara")
    parser.add_argument("--identidad", default=None, help="Nombre/ID de la persona a registrar")
    parser.add_argument("--muestras", type=int, default=40, help="Cantidad de muestras a capturar en tiempo real")
    parser.add_argument("--intervalo", type=int, default=5, help="Guardar una muestra cada N frames")
    parser.add_argument("--max-img-entrenamiento", type=int, default=None, help="Maximo de imagenes por identidad para entrenar; 0 usa todas")
    parser.add_argument("--sin-ventana", action="store_true", help="Ejecutar sin ventanas de OpenCV")
    parser.add_argument("--auto-entrenar", action="store_true", help="Entrenar automáticamente después de registrar")
    return parser


def obtener_fuente_defecto(configuracion: Dict[str, object]) -> str:
    """Obtiene la primera fuente configurada cuando el usuario no pasa --fuente."""
    fuentes = configuracion.get("camaras", {}).get("fuentes", [])
    if fuentes:
        fuente = fuentes[0]
        # Comentario clave: el YAML acepta valor para webcam/URL o ruta para videos grabados.
        return str(fuente.get("valor", fuente.get("url", fuente.get("path", 0))))
    return "0"


def configurar_uso_cpu(configuracion: Dict[str, object], multicamara: bool = False, video: bool = False) -> None:
    """Ajusta hilos de CPU para mejorar FPS sin ocupar todos los nucleos."""
    rendimiento = configuracion.get("rendimiento", {})
    nucleos = max(1, os.cpu_count() or 1)
    reservar = max(0, int(rendimiento.get("reservar_nucleos", 1)))
    hilos_auto = max(1, nucleos - reservar)

    if multicamara:
        hilos = int(rendimiento.get("hilos_cpu_multicamara", 0) or 0)
        if hilos <= 0:
            hilos = min(max(2, hilos_auto), 6)
    elif video:
        hilos = int(rendimiento.get("hilos_cpu_video", 0) or 0)
        if hilos <= 0:
            hilos = min(max(2, hilos_auto), 4)
    else:
        hilos = int(rendimiento.get("hilos_cpu", 0) or 0)
        if hilos <= 0:
            hilos = hilos_auto

    hilos = max(1, min(hilos, nucleos))
    try:
        cv2.setNumThreads(hilos)
    except cv2.error:
        pass

    try:  # pragma: no cover - torch puede no estar instalado en todos los entornos
        import torch

        torch.set_num_threads(hilos)
        torch.set_num_interop_threads(max(1, min(2, hilos // 2 or 1)))
    except Exception:
        pass


def ejecutar_entrenamiento(configuracion: dict) -> None:
    """Entrena SVM facial HoG y SVM Re-ID HSV con las capturas actuales."""
    modelos = entrenar_desde_capturas(configuracion)
    if modelos:
        print(f"[OK] Artefactos generados: {', '.join(modelos.keys())}")
    else:
        print("[AVISO] No se entrenó ningún modelo. Registra rostros con --modo registrar_rostro y/o cuerpo completo con --modo registrar_reid.")


def ejecutar_diagnostico(configuracion: dict) -> None:
    """Genera metricas del SVM facial y matriz de confusion."""
    metricas = diagnosticar_modelo_rostro(configuracion)
    carpeta_reportes = Path(configuracion["rutas"]["reportes"])
    print("[OK] Diagnostico facial generado")
    print(f"[OK] Accuracy: {metricas.get('accuracy', 0):.4f} | F1 macro: {metricas.get('f1_macro', 0):.4f}")
    print(f"[OK] Reporte: {carpeta_reportes / 'diagnostico_rostro.txt'}")
    print(f"[OK] Matriz CSV: {carpeta_reportes / 'matriz_confusion_rostro.csv'}")
    print(f"[OK] Matriz imagen: {carpeta_reportes / 'matriz_confusion_rostro.jpg'}")


def actualizar_reportes(configuracion: dict) -> None:
    """Regenera los archivos principales de la carpeta reportes."""
    carpeta_reportes = Path(configuracion["rutas"]["reportes"])
    carpeta_reportes.mkdir(parents=True, exist_ok=True)
    print("[INFO] Actualizando carpeta de reportes...")
    ejecutar_diagnostico(configuracion)
    print(f"[OK] Carpeta actualizada: {carpeta_reportes}")


def abrir_fuente(fuente: str):
    """Abre una imagen, video, URL o cámara según el valor recibido."""
    fuente = str(fuente)
    if fuente.isdigit() or fuente.startswith(("http://", "https://", "rtsp://")):
        # Comentario clave: aquí entran webcams locales, celulares por IP Webcam/DroidCam y RTSP.
        if fuente.isdigit():
            indice = int(fuente)
            captura = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
        else:
            captura = cv2.VideoCapture(fuente)
        optimizar_captura(captura)
        return captura, "video"

    ruta = Path(fuente)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la fuente: {fuente}")

    if ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        imagen = cargar_imagen_bgr(ruta)
        return imagen, "imagen"
    captura = cv2.VideoCapture(str(ruta))
    optimizar_captura(captura)
    return captura, "video"


def optimizar_captura(captura, ancho: Optional[int] = None, alto: Optional[int] = None, fps: Optional[int] = None) -> None:
    """Reduce latencia de lectura y fija resolucion cuando la camara lo permite."""
    if captura is None:
        return
    ajustes = [(cv2.CAP_PROP_BUFFERSIZE, 1)]
    if ancho:
        ajustes.append((cv2.CAP_PROP_FRAME_WIDTH, int(ancho)))
    if alto:
        ajustes.append((cv2.CAP_PROP_FRAME_HEIGHT, int(alto)))
    if fps:
        ajustes.append((cv2.CAP_PROP_FPS, int(fps)))

    for propiedad, valor in ajustes:
        try:
            captura.set(propiedad, valor)
        except cv2.error:
            pass


def obtener_fuentes_multicamara(configuracion: Dict[str, object], max_camaras: int = 4) -> List[Tuple[str, str]]:
    """Obtiene hasta cuatro fuentes; si faltan, completa con indices locales."""
    fuentes_config = configuracion.get("camaras", {}).get("fuentes", [])
    fuentes: List[Tuple[str, str]] = []
    valores_vistos = set()

    for indice, fuente in enumerate(fuentes_config):
        if len(fuentes) >= max_camaras:
            break
        valor = str(fuente.get("valor", fuente.get("url", fuente.get("path", indice))))
        if valor in valores_vistos:
            continue
        camara_id = str(fuente.get("id", f"camara_{len(fuentes) + 1:02d}"))
        fuentes.append((camara_id, valor))
        valores_vistos.add(valor)

    indice_local = 0
    while len(fuentes) < max_camaras:
        valor = str(indice_local)
        if valor not in valores_vistos:
            fuentes.append((f"camara_{len(fuentes) + 1:02d}", valor))
            valores_vistos.add(valor)
        indice_local += 1
    return fuentes[:max_camaras]


@dataclass
class LectorCamara:
    """Mantiene el ultimo frame disponible de una camara sin acumular cola."""

    camara_id: str
    fuente: str
    captura: object
    activa: bool = False
    frame: Optional[np.ndarray] = None
    frame_numero: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._hilo: Optional[threading.Thread] = None

    def iniciar(self) -> None:
        if not self.captura or not self.captura.isOpened():
            self.error = "no disponible"
            return
        self.activa = True
        self._hilo = threading.Thread(target=self._leer, name=f"lector-{self.camara_id}", daemon=True)
        self._hilo.start()

    def _leer(self) -> None:
        fallos = 0
        while self.activa:
            ok, frame = self.captura.read()
            if not ok:
                fallos += 1
                self.error = "sin senal"
                if fallos >= 30:
                    time.sleep(0.15)
                continue
            fallos = 0
            with self._lock:
                self.frame = frame
                self.frame_numero += 1
                self.error = ""

    def obtener_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self.frame is None:
                return None, self.frame_numero
            return self.frame.copy(), self.frame_numero

    def detener(self) -> None:
        self.activa = False
        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=1.0)
        if self.captura:
            self.captura.release()


def guardar_log_predicciones(ruta_csv: Path, frame_numero: int, resultados: Iterable[ResultadoIdentidad], fps: float) -> None:
    """Guarda trazabilidad de predicciones por frame."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_csv.exists()
    campos = ["frame", "identidad", "metodo", "score", "bbox", "fps", "detalle", "ranking"]
    with ruta_csv.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            escritor.writeheader()
        for resultado in resultados:
            # Comentario clave: estos logs sirven como evidencia para métricas y defensa del laboratorio.
            escritor.writerow(
                {
                    "frame": frame_numero,
                    "identidad": resultado.identidad,
                    "metodo": resultado.metodo,
                    "score": f"{resultado.score:.4f}",
                    "bbox": list(resultado.caja),
                    "fps": f"{fps:.2f}",
                    "detalle": resultado.detalle,
                    "ranking": resultado.ranking,
                }
            )


def guardar_log_predicciones_camara(
    ruta_csv: Path,
    camara_id: str,
    frame_numero: int,
    resultados: Iterable[ResultadoIdentidad],
    fps: float,
) -> None:
    """Guarda predicciones multicamara con el identificador de fuente."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta_csv.exists()
    campos = ["camara", "frame", "identidad", "metodo", "score", "bbox", "fps", "detalle", "ranking"]
    with ruta_csv.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        if not existe:
            escritor.writeheader()
        for resultado in resultados:
            escritor.writerow(
                {
                    "camara": camara_id,
                    "frame": frame_numero,
                    "identidad": resultado.identidad,
                    "metodo": resultado.metodo,
                    "score": f"{resultado.score:.4f}",
                    "bbox": list(resultado.caja),
                    "fps": f"{fps:.2f}",
                    "detalle": resultado.detalle,
                    "ranking": resultado.ranking,
                }
            )


def redimensionar_para_inferencia(frame: np.ndarray, ancho_maximo: int) -> np.ndarray:
    """Reduce frames grandes antes de detectar para sostener mejor FPS."""
    if ancho_maximo <= 0:
        return frame
    alto, ancho = frame.shape[:2]
    if ancho <= ancho_maximo:
        return frame
    escala = ancho_maximo / float(ancho)
    nuevo_tamano = (ancho_maximo, max(1, int(alto * escala)))
    return cv2.resize(frame, nuevo_tamano, interpolation=cv2.INTER_AREA)


def es_archivo_video(fuente: str) -> bool:
    """Indica si la fuente apunta a un archivo de video local."""
    ruta = Path(str(fuente))
    if not ruta.exists() or not ruta.is_file():
        return False
    return ruta.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def dibujar_barra_video(frame: np.ndarray, fps: float, frame_numero: int) -> np.ndarray:
    """Dibuja una barra compacta para videos cargados desde archivo."""
    salida = frame.copy()
    texto = f"Video | FPS inf {fps:.1f} | frame {frame_numero} | q salir"
    cv2.rectangle(salida, (0, 0), (salida.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(salida, texto, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return salida


def dibujar_barra_captura_roi(frame: np.ndarray, fps: float, frame_numero: int, conteo: Dict[str, int]) -> np.ndarray:
    """Dibuja barra de estado para la captura manual de rostros v2."""
    salida = frame.copy()
    resumen = ", ".join(f"{k}:{v}" for k, v in sorted(conteo.items())) or "0"
    texto = f"Captura ROI rostros v2 | FPS {fps:.1f} | frame {frame_numero} | guardadas {resumen} | q salir"
    cv2.rectangle(salida, (0, 0), (salida.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(salida, texto[:150], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return salida


def ajustar_a_celda(frame: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Encaja un frame en una celda fija sin deformarlo."""
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    escala = min(ancho / max(1, w), alto / max(1, h))
    nuevo_w = max(1, int(w * escala))
    nuevo_h = max(1, int(h * escala))
    interpolacion = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
    redimensionado = cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=interpolacion)
    x = (ancho - nuevo_w) // 2
    y = (alto - nuevo_h) // 2
    lienzo[y : y + nuevo_h, x : x + nuevo_w] = redimensionado
    return lienzo


def obtener_tamano_pantalla(defecto: tuple[int, int] = (1280, 720)) -> tuple[int, int]:
    """Obtiene el tamano de pantalla para modo video completo."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        ancho = int(root.winfo_screenwidth())
        alto = int(root.winfo_screenheight())
        root.destroy()
        if ancho > 0 and alto > 0:
            return ancho, alto
    except Exception:
        pass
    return defecto


def escalar_caja(caja: tuple[int, int, int, int], escala_x: float, escala_y: float) -> tuple[int, int, int, int]:
    """Escala una caja desde el frame de inferencia al frame visual."""
    x1, y1, x2, y2 = caja
    return (
        int(round(x1 * escala_x)),
        int(round(y1 * escala_y)),
        int(round(x2 * escala_x)),
        int(round(y2 * escala_y)),
    )


def escalar_resultados(
    resultados: Iterable[ResultadoIdentidad],
    origen_shape: tuple[int, int, int],
    destino_shape: tuple[int, int, int],
) -> List[ResultadoIdentidad]:
    """Convierte cajas de resultados entre dimensiones sin recalcular inferencia."""
    origen_h, origen_w = origen_shape[:2]
    destino_h, destino_w = destino_shape[:2]
    escala_x = destino_w / max(1, origen_w)
    escala_y = destino_h / max(1, origen_h)
    escalados: List[ResultadoIdentidad] = []
    for resultado in resultados:
        caja_rostro = (
            escalar_caja(resultado.caja_rostro, escala_x, escala_y)
            if resultado.caja_rostro is not None
            else None
        )
        escalados.append(
            replace(
                resultado,
                caja=escalar_caja(resultado.caja, escala_x, escala_y),
                caja_rostro=caja_rostro,
            )
        )
    return escalados


def nombre_seguro_dataset(texto: str) -> str:
    """Normaliza nombres para crear carpetas de dataset en Windows."""
    limpio = re.sub(r'[<>:"/\\|?*]+', "_", str(texto)).strip()
    limpio = re.sub(r"\s+", " ", limpio)
    return limpio or "desconocido"


def recortar_caja_con_margen(imagen: np.ndarray, caja: tuple[int, int, int, int], margen: float = 0.22) -> np.ndarray:
    """Recorta una caja con margen alrededor para no dejar el rostro demasiado pegado."""
    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in caja]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    extra_x = int(w * margen)
    extra_y = int(h * margen)
    x1 = max(0, x1 - extra_x)
    y1 = max(0, y1 - extra_y)
    x2 = min(ancho, x2 + extra_x)
    y2 = min(alto, y2 + extra_y)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=imagen.dtype)
    return imagen[y1:y2, x1:x2]


def detectar_rostros_por_persona(
    frame: np.ndarray,
    detector_personas: DetectorPersonasYOLO,
    detector_rostros: DetectorRostros,
    configuracion: dict,
) -> List[tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]]:
    """Detecta personas con YOLO y ROI facial con Haar dentro de cada persona."""
    rostros_encontrados: List[tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]] = []
    umbrales = configuracion.get("umbrales", {})
    config_zoom = configuracion.get("rostro_en_persona", {})
    tamano_minimo = int(umbrales.get("tamano_minimo_rostro", 40))
    clase_persona = int(configuracion.get("yolo", {}).get("clase_persona", 0))

    for persona in detector_personas.detectar_personas(frame, clase_persona=clase_persona):
        rostro = detector_rostros.detectar_rostro_principal(persona.roi)
        origen = "ROI rostro"

        if rostro is None and bool(config_zoom.get("usar_zoom_si_no_detecta", True)):
            roi_persona = persona.roi
            alto, ancho = roi_persona.shape[:2]
            porcentaje_superior = max(0.10, min(1.0, float(config_zoom.get("porcentaje_superior", 0.45))))
            y2_superior = max(1, min(alto, int(alto * porcentaje_superior)))
            zona_superior = roi_persona[:y2_superior, :]
            factor_zoom = max(1.0, min(6.0, float(config_zoom.get("factor_zoom", 3.0))))
            tamano_minimo_zoom = int(config_zoom.get("tamano_minimo_zoom", 18))

            zona_zoom = cv2.resize(
                zona_superior,
                (max(1, int(zona_superior.shape[1] * factor_zoom)), max(1, int(zona_superior.shape[0] * factor_zoom))),
                interpolation=cv2.INTER_CUBIC,
            )
            rostros_zoom = detector_rostros.detectar_rostros(zona_zoom, tamano_minimo=tamano_minimo_zoom)
            if rostros_zoom:
                rostro_zoom = rostros_zoom[0]
                zx1, zy1, zx2, zy2 = rostro_zoom.caja
                x1 = max(0, min(int(zx1 / factor_zoom), ancho - 1))
                y1 = max(0, min(int(zy1 / factor_zoom), alto - 1))
                x2 = max(x1 + 1, min(int(zx2 / factor_zoom), ancho))
                y2 = max(y1 + 1, min(int(zy2 / factor_zoom), alto))
                rostro = replace(rostro_zoom, caja=(x1, y1, x2, y2), roi=roi_persona[y1:y2, x1:x2])
                origen = "ROI rostro zoom"

        if rostro is None:
            continue

        px1, py1, _, _ = persona.caja
        rx1, ry1, rx2, ry2 = rostro.caja
        caja_rostro = (px1 + rx1, py1 + ry1, px1 + rx2, py1 + ry2)
        rostros_encontrados.append((caja_rostro, persona.caja, origen))

    return rostros_encontrados


def capturar_roi_rostros_desde_video(
    configuracion: dict,
    fuente: str,
    intervalo_frames: int = 5,
    max_por_identidad: int = 50,
    mostrar_ventana: bool = True,
) -> Dict[str, int]:
    """Recorre un video, detecta ROI de rostro y lo guarda directamente en datos/rostros_v2."""
    configurar_uso_cpu(configuracion, multicamara=False)
    entrada, tipo = abrir_fuente(fuente)
    if tipo != "video":
        raise ValueError("Esta opcion espera una fuente de video, no una imagen.")
    captura = entrada
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {fuente}")

    rutas = configuracion.get("rutas", {})
    carpeta_rostros = Path(str(rutas.get("rostros_v2", Path(str(rutas.get("datos", "datos"))) / "rostros_v2")))
    carpeta_rostros.mkdir(parents=True, exist_ok=True)
    carpeta_registros = Path(configuracion["rutas"]["registros"])
    carpeta_registros.mkdir(parents=True, exist_ok=True)
    ruta_metadata = carpeta_registros / "capturas_rostros_v2_video.csv"
    existe_metadata = ruta_metadata.exists()
    campos = ["fecha", "video", "frame", "id_temporal", "caja_rostro", "ruta"]

    video_config = configuracion.get("video", {})
    respetar_resolucion_original = bool(video_config.get("respetar_resolucion_original", True))
    vista_original_estricta = bool(video_config.get("vista_original_estricta", False))
    pantalla_completa_video = bool(video_config.get("pantalla_completa", True))
    ancho_video = int(video_config.get("ancho_proceso", 800))
    ancho_vista_video = int(video_config.get("ancho_vista", 1280))
    alto_vista_video = int(video_config.get("alto_vista", 720))
    ancho_pantalla, alto_pantalla = obtener_tamano_pantalla((ancho_vista_video, alto_vista_video))
    respetar_fps_video = bool(video_config.get("respetar_fps", True))
    intervalo_frames = max(1, int(intervalo_frames))
    max_por_identidad = max(0, int(max_por_identidad))
    mostrar = bool(mostrar_ventana and configuracion.get("ejecucion", {}).get("mostrar_ventana", True))
    conteo: Dict[str, int] = {}
    ultimo_guardado: Dict[str, int] = {}
    frame_numero = 0
    nombre_video = Path(str(fuente)).stem if not str(fuente).isdigit() else f"camara_{fuente}"
    yolo_config = configuracion.get("yolo", {})
    tamano_yolo_video = int(video_config.get("tamano_yolo", yolo_config.get("tamano_imagen", 640)))
    detector_personas = DetectorPersonasYOLO(
        pesos=str(yolo_config.get("pesos", "modelos/yolov8n.pt")),
        confianza=float(yolo_config.get("confianza", 0.40)),
        tamano_imagen=tamano_yolo_video if tamano_yolo_video > 0 else int(yolo_config.get("tamano_imagen", 640)),
        dispositivo=str(yolo_config.get("dispositivo", configuracion.get("ejecucion", {}).get("dispositivo", "cpu"))),
    )
    detector_rostros = DetectorRostros()
    rastreador = RastreadorSimple(distancia_maxima=90.0, max_frames_perdidos=8)
    fps_video_origen = float(captura.get(cv2.CAP_PROP_FPS) or 0.0)
    ancho_origen = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto_origen = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps_video_origen <= 1.0 or fps_video_origen > 120.0:
        fps_video_origen = float(video_config.get("fps_por_defecto", 30))
    duracion_frame_video = 1.0 / max(1.0, fps_video_origen)
    inicio_reproduccion = time.time()
    ultimo_tiempo = time.time()
    nombre_ventana = "Captura ROI rostros v2"

    print("[INFO] Captura ROI de rostros iniciada. Usa YOLO para persona + Haar para rostro; no usa SVM ni predice identidad.")
    print("[INFO] Las capturas se guardan en datos/rostros_v2 para clasificacion manual.")
    print(f"[INFO] Video origen: {ancho_origen}x{alto_origen} @ {fps_video_origen:.2f} FPS")
    print(f"[INFO] Fuente: {fuente} | guardar cada {intervalo_frames} frames | max/id temporal: {'sin limite' if max_por_identidad <= 0 else max_por_identidad}")

    if mostrar:
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        if pantalla_completa_video:
            try:
                cv2.setWindowProperty(nombre_ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except cv2.error:
                pass

    with ruta_metadata.open("a", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.DictWriter(archivo_csv, fieldnames=campos)
        if not existe_metadata:
            escritor.writeheader()

        while True:
            ok, frame = captura.read()
            if not ok:
                break
            frame_numero += 1

            ahora = time.time()
            fps_visual = 1.0 / max(0.001, ahora - ultimo_tiempo)
            ultimo_tiempo = ahora

            frame_proceso = redimensionar_para_inferencia(frame, ancho_video)
            detecciones_rostro = detectar_rostros_por_persona(frame_proceso, detector_personas, detector_rostros, configuracion)
            escala_x = frame.shape[1] / max(1, frame_proceso.shape[1])
            escala_y = frame.shape[0] / max(1, frame_proceso.shape[0])
            cajas_rostro = [escalar_caja(caja_rostro, escala_x, escala_y) for caja_rostro, _, _ in detecciones_rostro]
            cajas_persona = [escalar_caja(caja_persona, escala_x, escala_y) for _, caja_persona, _ in detecciones_rostro]
            detalles_rostro = [detalle for _, _, detalle in detecciones_rostro]
            pistas = rastreador.actualizar(cajas_rostro)
            ids_por_caja = {
                pista.caja: pista.id_pista
                for pista in pistas.values()
                if pista.frames_perdidos == 0
            }

            salida = frame.copy()
            for indice, caja in enumerate(cajas_rostro):
                id_pista = ids_por_caja.get(caja)
                if id_pista is None:
                    continue

                id_temporal = f"rostro_{id_pista:03d}"
                puede_guardar_por_limite = max_por_identidad <= 0 or conteo.get(id_temporal, 0) < max_por_identidad
                puede_guardar_por_intervalo = frame_numero - ultimo_guardado.get(id_temporal, -intervalo_frames) >= intervalo_frames
                guardado = False
                if puede_guardar_por_limite and puede_guardar_por_intervalo:
                    rostro = recortar_caja_con_margen(frame, caja)
                    if rostro.size > 0:
                        conteo[id_temporal] = conteo.get(id_temporal, 0) + 1
                        ultimo_guardado[id_temporal] = frame_numero
                        nombre_archivo = f"{id_temporal}_{nombre_seguro_dataset(nombre_video)}_f{frame_numero:06d}_{conteo[id_temporal]:04d}.jpg"
                        ruta = carpeta_rostros / nombre_archivo
                        guardar_imagen_bgr(ruta, rostro)
                        escritor.writerow(
                            {
                                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "video": str(fuente),
                                "frame": frame_numero,
                                "id_temporal": id_temporal,
                                "caja_rostro": list(caja),
                                "ruta": str(ruta),
                            }
                        )
                        guardado = True

                x1, y1, x2, y2 = caja
                color = (80, 255, 120) if guardado else (255, 210, 80)
                if indice < len(cajas_persona):
                    px1, py1, px2, py2 = cajas_persona[indice]
                    cv2.rectangle(salida, (px1, py1), (px2, py2), (90, 90, 90), 1)
                origen = detalles_rostro[indice] if indice < len(detalles_rostro) else "ROI rostro"
                texto_caja = f"{id_temporal} {'guardado' if guardado else origen}"
                cv2.rectangle(salida, (x1, y1), (x2, y2), color, 2)
                cv2.putText(salida, texto_caja, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if mostrar:
                salida = dibujar_barra_captura_roi(salida, fps_visual, frame_numero, conteo)
                if pantalla_completa_video:
                    salida = ajustar_a_celda(salida, ancho_pantalla, alto_pantalla)
                elif not respetar_resolucion_original or not vista_original_estricta:
                    salida = ajustar_a_celda(salida, ancho_vista_video, alto_vista_video)
                cv2.imshow(nombre_ventana, salida)
                espera_ms = 1
                if respetar_fps_video:
                    objetivo = inicio_reproduccion + frame_numero * duracion_frame_video
                    espera = objetivo - time.time()
                    espera_ms = max(1, int(espera * 1000)) if espera > 0 else 1
                if cv2.waitKey(espera_ms) & 0xFF == ord("q"):
                    break

    captura.release()
    if mostrar:
        cv2.destroyWindow(nombre_ventana)

    print(f"[OK] Captura finalizada. Conteo: {conteo}")
    print(f"[OK] Carpeta destino: {carpeta_rostros}")
    print(f"[OK] Metadata: {ruta_metadata}")
    return conteo


def dibujar_etiqueta_camara(frame: np.ndarray, camara_id: str, fuente: str, fps: float, frame_numero: int) -> np.ndarray:
    """Dibuja identificacion compacta para cada celda de la cuadricula."""
    salida = frame.copy()
    texto = f"{camara_id} | {fuente} | FPS inf {fps:.1f} | frame {frame_numero}"
    cv2.rectangle(salida, (0, 0), (salida.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(salida, texto[:95], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return salida


def crear_placeholder_camara(camara_id: str, fuente: str, ancho: int, alto: int, estado: str) -> np.ndarray:
    """Crea una celda visible para camaras no abiertas o sin senal."""
    imagen = np.full((alto, ancho, 3), (28, 30, 34), dtype=np.uint8)
    cv2.putText(imagen, camara_id, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(imagen, str(fuente)[:70], (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(imagen, estado or "esperando senal", (24, alto // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
    return imagen


def componer_cuadricula(vistas: List[np.ndarray], ancho: int, alto: int) -> np.ndarray:
    """Compone cuatro vistas en una cuadricula 2x2."""
    while len(vistas) < 4:
        vistas.append(np.zeros((alto, ancho, 3), dtype=np.uint8))
    fila_1 = np.hstack(vistas[:2])
    fila_2 = np.hstack(vistas[2:4])
    return np.vstack([fila_1, fila_2])


def ejecutar_inferencia_multicamara(configuracion: dict, fuentes: List[Tuple[str, str]]) -> None:
    """Ejecuta inferencia de hasta cuatro camaras en una pantalla 2x2."""
    configurar_uso_cpu(configuracion, multicamara=True)
    multi = configuracion.get("multicamara", {})
    ancho_proceso = int(multi.get("ancho_proceso", 640))
    ancho_celda = int(multi.get("ancho_celda", 640))
    alto_celda = int(multi.get("alto_celda", 360))
    intervalo = max(0.01, int(multi.get("intervalo_inferencia_ms", 120)) / 1000.0)
    tamano_yolo = int(multi.get("tamano_yolo", 416))
    if tamano_yolo > 0:
        yolo = configuracion.setdefault("yolo", {})
        yolo["tamano_imagen"] = min(int(yolo.get("tamano_imagen", tamano_yolo)), tamano_yolo)

    carpeta_registros = Path(configuracion["rutas"]["registros"])
    carpeta_registros.mkdir(parents=True, exist_ok=True)

    lectores: List[LectorCamara] = []
    for camara_id, fuente in fuentes[:4]:
        try:
            entrada, tipo = abrir_fuente(fuente)
        except Exception as exc:
            print(f"[AVISO] {camara_id}: no se pudo abrir {fuente}: {exc}")
            lectores.append(LectorCamara(camara_id=camara_id, fuente=str(fuente), captura=None, error="no disponible"))
            continue
        if tipo != "video":
            print(f"[AVISO] {camara_id}: la fuente no es video/camara: {fuente}")
            lectores.append(LectorCamara(camara_id=camara_id, fuente=str(fuente), captura=None, error="no es video"))
            continue
        optimizar_captura(entrada, ancho=ancho_proceso, alto=alto_celda)
        lector = LectorCamara(camara_id=camara_id, fuente=str(fuente), captura=entrada)
        lector.iniciar()
        lectores.append(lector)

    if not lectores:
        raise RuntimeError("No hay fuentes de video para la vista multicamara.")
    if not any(lector.activa for lector in lectores):
        for lector in lectores:
            lector.detener()
        raise RuntimeError("No se pudo abrir ninguna camara para la vista multicamara.")

    motor = MotorInferencia(configuracion)
    motor.cargar_modelos()
    nombre_ventana = "Re-ID PC Python | 4 camaras"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    resultados_por_camara: Dict[str, List[ResultadoIdentidad]] = {lector.camara_id: [] for lector in lectores}
    fps_por_camara: Dict[str, float] = {lector.camara_id: 0.0 for lector in lectores}
    ultimo_proceso: Dict[str, float] = {lector.camara_id: 0.0 for lector in lectores}
    indice_turno = 0
    total_procesados = 0
    ruta_log = carpeta_registros / "predicciones_multicamara.csv"

    print("[INFO] Inferencia multicamara iniciada. Presiona 'q' para salir, 'd' para diagnostico.")
    try:
        while True:
            ahora = time.time()
            for _ in range(len(lectores)):
                lector = lectores[indice_turno % len(lectores)]
                indice_turno += 1
                frame, frame_numero = lector.obtener_frame()
                if frame is None or ahora - ultimo_proceso[lector.camara_id] < intervalo:
                    continue

                frame_proceso = redimensionar_para_inferencia(frame, ancho_proceso)
                inicio = time.time()
                resultados = motor.procesar_frame(frame_proceso)
                duracion = max(0.001, time.time() - inicio)
                fps = 1.0 / duracion
                resultados_por_camara[lector.camara_id] = resultados
                fps_por_camara[lector.camara_id] = fps
                ultimo_proceso[lector.camara_id] = time.time()
                total_procesados += 1
                guardar_log_predicciones_camara(ruta_log, lector.camara_id, frame_numero, resultados, fps)
                break

            vistas: List[np.ndarray] = []
            for lector in lectores[:4]:
                frame, frame_numero = lector.obtener_frame()
                if frame is None:
                    vistas.append(crear_placeholder_camara(lector.camara_id, lector.fuente, ancho_celda, alto_celda, lector.error))
                    continue
                frame_base = redimensionar_para_inferencia(frame, ancho_proceso)
                salida = dibujar_resultados(
                    frame_base,
                    resultados_por_camara.get(lector.camara_id, []),
                    modo_cajas=str(configuracion.get("visualizacion", {}).get("modo_cajas", "ambas")),
                )
                salida = dibujar_etiqueta_camara(
                    salida,
                    lector.camara_id,
                    lector.fuente,
                    fps_por_camara.get(lector.camara_id, 0.0),
                    frame_numero,
                )
                vistas.append(ajustar_a_celda(salida, ancho_celda, alto_celda))

            cv2.imshow(nombre_ventana, componer_cuadricula(vistas, ancho_celda, alto_celda))
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("d"):
                print("[INFO] Generando diagnostico facial...")
                try:
                    ejecutar_diagnostico(configuracion)
                except Exception as exc:
                    print(f"[AVISO] No se pudo generar diagnostico: {exc}")
            if tecla == ord("q"):
                break
    finally:
        for lector in lectores:
            lector.detener()
        cv2.destroyAllWindows()

    print(f"[OK] Inferencia multicamara finalizada. Ciclos de inferencia: {total_procesados}")
    print(f"[OK] Logs: {ruta_log}")


def ejecutar_inferencia(configuracion: dict, fuente: str) -> None:
    """Ejecuta inferencia sobre imagen, video, URL o cámara."""
    entrada, tipo = abrir_fuente(fuente)
    fuente_es_video = tipo == "video" and es_archivo_video(fuente)
    configurar_uso_cpu(configuracion, multicamara=False, video=(tipo == "video"))
    if fuente_es_video:
        video_config_previa = configuracion.get("video", {})
        tamano_yolo_video_previo = int(video_config_previa.get("tamano_yolo", 416))
        if tamano_yolo_video_previo > 0:
            yolo = configuracion.setdefault("yolo", {})
            yolo["tamano_imagen"] = min(int(yolo.get("tamano_imagen", tamano_yolo_video_previo)), tamano_yolo_video_previo)
    motor = MotorInferencia(configuracion)
    motor.cargar_modelos()
    carpeta_salida = Path(configuracion["rutas"]["salidas"])
    carpeta_registros = Path(configuracion["rutas"]["registros"])
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    carpeta_registros.mkdir(parents=True, exist_ok=True)

    if tipo == "imagen":
        resultados = motor.procesar_frame(entrada)
        salida = dibujar_resultados(
            entrada,
            resultados,
            modo_cajas=str(configuracion.get("visualizacion", {}).get("modo_cajas", "ambas")),
        )
        ruta_salida = carpeta_salida / "resultado_imagen.jpg"
        cv2.imwrite(str(ruta_salida), salida)
        guardar_log_predicciones(carpeta_registros / "predicciones.csv", 1, resultados, 0.0)
        print(f"[OK] Detecciones procesadas: {len(resultados)}")
        print(f"[OK] Resultado guardado en: {ruta_salida}")
        return

    captura = entrada
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    total_frames_video = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if fuente_es_video else 0
    video_config = configuracion.get("video", {})
    respetar_resolucion_original = bool(video_config.get("respetar_resolucion_original", True))
    vista_original_estricta = bool(video_config.get("vista_original_estricta", False))
    pantalla_completa_video = bool(video_config.get("pantalla_completa", True))
    panel_separado_video = bool(video_config.get("panel_separado", True))
    alto_panel_video = int(video_config.get("alto_panel", 720))
    ancho_video = int(video_config.get("ancho_proceso", 960))
    ancho_vista_video = int(video_config.get("ancho_vista", 960))
    alto_vista_video = int(video_config.get("alto_vista", 540))
    ancho_pantalla, alto_pantalla = obtener_tamano_pantalla((ancho_vista_video, alto_vista_video))
    procesar_cada = max(1, int(video_config.get("procesar_cada_n_frames", 2)))
    modo_fluido_video = bool(video_config.get("modo_fluido", True))
    max_desfase_roi = max(0, int(video_config.get("max_desfase_roi_frames", 0)))
    respetar_fps_video = bool(video_config.get("respetar_fps", True))
    saltar_atrasados = bool(video_config.get("saltar_frames_atrasados", True))
    inferencia_async_video = bool(fuente_es_video and modo_fluido_video)

    frame_numero = 0
    ultimo_tiempo = time.time()
    inicio_reproduccion = time.time()
    fps_video_origen = float(captura.get(cv2.CAP_PROP_FPS) or 0.0)
    ancho_origen = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto_origen = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps_video_origen <= 1.0 or fps_video_origen > 120.0:
        fps_video_origen = float(video_config.get("fps_por_defecto", 30))
    duracion_frame_video = 1.0 / max(1.0, fps_video_origen)
    if fuente_es_video:
        if respetar_resolucion_original and vista_original_estricta:
            modo_vista = "original/autosize"
        elif pantalla_completa_video:
            modo_vista = f"pantalla completa sin deformar {ancho_pantalla}x{alto_pantalla}"
        elif respetar_resolucion_original:
            modo_vista = f"original -> ajustado sin deformar {ancho_vista_video}x{alto_vista_video}"
        else:
            modo_vista = f"{ancho_vista_video}x{alto_vista_video}"
        print(f"[INFO] Video origen: {ancho_origen}x{alto_origen} @ {fps_video_origen:.2f} FPS | vista: {modo_vista}")
    resultados_ultimo: List[ResultadoIdentidad] = []
    frame_resultados_ultimo = 0
    fps_ultimo = 0.0
    lock_inferencia = threading.Lock()
    hilo_inferencia: Optional[threading.Thread] = None
    inferencia_ocupada = False
    frame_pendiente: Optional[np.ndarray] = None
    numero_frame_pendiente = 0
    evento_inferencia = threading.Event()
    detener_inferencia = threading.Event()
    mostrar = bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True))
    usar_panel = bool(configuracion.get("panel_control", {}).get("activo", True))
    nombre_ventana = "Re-ID Video" if fuente_es_video else "Re-ID PC Python"
    ventana_original = bool(fuente_es_video and respetar_resolucion_original and vista_original_estricta)
    usar_panel_separado = bool(fuente_es_video and usar_panel and panel_separado_video)
    nombre_panel = "Panel Re-ID Video" if usar_panel_separado else nombre_ventana
    panel = (
        PanelControlInferencia(nombre_panel, configuracion, ventana_autosize=False if usar_panel_separado else ventana_original)
        if mostrar and usar_panel
        else None
    )
    if mostrar:
        if usar_panel_separado or panel is None:
            cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        if usar_panel_separado and hasattr(cv2, "WND_PROP_TOPMOST"):
            try:
                cv2.setWindowProperty(nombre_panel, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
        if fuente_es_video and pantalla_completa_video:
            try:
                cv2.setWindowProperty(nombre_ventana, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except cv2.error:
                pass

    def lanzar_inferencia_video(frame_modelo: np.ndarray, numero_frame: int) -> bool:
        """Entrega un frame al trabajador solo cuando no hay inferencia pendiente."""
        nonlocal frame_pendiente, numero_frame_pendiente

        with lock_inferencia:
            if inferencia_ocupada or frame_pendiente is not None:
                return False
            frame_pendiente = frame_modelo.copy()
            numero_frame_pendiente = numero_frame
            evento_inferencia.set()
            return True

    def trabajador_inferencia_video() -> None:
        """Procesa el ultimo frame disponible y descarta pendientes antiguos."""
        nonlocal resultados_ultimo, frame_resultados_ultimo, fps_ultimo, inferencia_ocupada
        nonlocal frame_pendiente, numero_frame_pendiente

        while not detener_inferencia.is_set():
            evento_inferencia.wait(0.05)
            if detener_inferencia.is_set():
                break

            with lock_inferencia:
                if frame_pendiente is None:
                    evento_inferencia.clear()
                    continue
                frame_modelo = frame_pendiente
                numero_frame = numero_frame_pendiente
                frame_pendiente = None
                evento_inferencia.clear()
                inferencia_ocupada = True

            try:
                inicio = time.time()
                resultados = motor.procesar_frame(frame_modelo)
                fps_inferencia = 1.0 / max(0.001, time.time() - inicio)
                guardar_log_predicciones(carpeta_registros / "predicciones.csv", numero_frame, resultados, fps_inferencia)
                with lock_inferencia:
                    if resultados or not resultados_ultimo:
                        resultados_ultimo = resultados
                        frame_resultados_ultimo = numero_frame
                    fps_ultimo = fps_inferencia
            except Exception as exc:
                print(f"[AVISO] Inferencia de video fallida: {exc}")
            finally:
                with lock_inferencia:
                    inferencia_ocupada = False

    if inferencia_async_video:
        hilo_inferencia = threading.Thread(target=trabajador_inferencia_video, name="inferencia-video", daemon=True)
        hilo_inferencia.start()

    while True:
        ok, frame = captura.read()
        if not ok:
            break
        frame_numero += 1

        ahora = time.time()
        fps = 1.0 / max(0.001, ahora - ultimo_tiempo)
        ultimo_tiempo = ahora

        frame_proceso = redimensionar_para_inferencia(frame, ancho_video) if fuente_es_video else frame
        debe_procesar = not fuente_es_video or not resultados_ultimo or (frame_numero - 1) % procesar_cada == 0
        if inferencia_async_video:
            if debe_procesar:
                lanzar_inferencia_video(frame_proceso, frame_numero)
        elif debe_procesar:
            inicio = time.time()
            resultados_nuevos = motor.procesar_frame(frame_proceso)
            if resultados_nuevos or not resultados_ultimo:
                resultados_ultimo = resultados_nuevos
                frame_resultados_ultimo = frame_numero
            fps_ultimo = 1.0 / max(0.001, time.time() - inicio)
            guardar_log_predicciones(carpeta_registros / "predicciones.csv", frame_numero, resultados_nuevos, fps_ultimo)

        with lock_inferencia:
            resultados = list(resultados_ultimo)
            frame_resultados = frame_resultados_ultimo
            fps_mostrar = fps if fuente_es_video and modo_fluido_video else (fps_ultimo if fuente_es_video else fps)
        if fuente_es_video and max_desfase_roi > 0 and frame_resultados > 0 and frame_numero - frame_resultados > max_desfase_roi:
            resultados = []
        frame_dibujo = frame
        resultados_dibujo = resultados
        if fuente_es_video:
            resultados_dibujo = escalar_resultados(resultados, frame_proceso.shape, frame.shape)
        else:
            frame_dibujo = frame_proceso
        salida = dibujar_resultados(
            frame_dibujo,
            resultados_dibujo,
            modo_cajas=str(configuracion.get("visualizacion", {}).get("modo_cajas", "ambas")),
        )
        if fuente_es_video and pantalla_completa_video:
            salida = ajustar_a_celda(salida, ancho_pantalla, alto_pantalla)
        elif fuente_es_video and (not respetar_resolucion_original or not vista_original_estricta):
            salida = ajustar_a_celda(salida, ancho_vista_video, alto_vista_video)
        if fuente_es_video and panel is None:
            salida = dibujar_barra_video(salida, fps_mostrar, frame_numero)

        # Comentario clave: la ventana permite validar visualmente el prototipo durante la exposición.
        if mostrar:
            if panel and usar_panel_separado:
                vista = salida
                cv2.imshow(nombre_ventana, vista)
                vista_panel = panel.construir_panel(resultados_dibujo, fps_mostrar, frame_numero, alto_visible=alto_panel_video)
                cv2.imshow(nombre_panel, vista_panel)
            else:
                vista = panel.construir_vista(salida, resultados_dibujo, fps_mostrar, frame_numero) if panel else salida
                cv2.imshow(nombre_ventana, vista)
            espera_ms = 1
            if fuente_es_video and respetar_fps_video:
                objetivo = inicio_reproduccion + frame_numero * duracion_frame_video
                espera = objetivo - time.time()
                espera_ms = max(1, int(espera * 1000)) if espera > 0 else 1
            tecla = cv2.waitKey(espera_ms) & 0xFF
            if tecla == ord("d"):
                print("[INFO] Generando diagnostico facial...")
                try:
                    ejecutar_diagnostico(configuracion)
                    if panel:
                        panel.recargar_diagnostico()
                except Exception as exc:
                    print(f"[AVISO] No se pudo generar diagnostico: {exc}")
            if tecla == ord("q"):
                break

            if fuente_es_video and respetar_fps_video and saltar_atrasados:
                objetivo = inicio_reproduccion + frame_numero * duracion_frame_video
                retraso = time.time() - objetivo
                frames_a_saltar = max(0, int(retraso / duracion_frame_video))
                frames_a_saltar = min(frames_a_saltar, max(1, procesar_cada * 4))
                if total_frames_video > 0:
                    frame_posicion = int(captura.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    reserva_final = max(2, procesar_cada * 2)
                    frames_restantes = max(0, total_frames_video - frame_posicion)
                    frames_a_saltar = min(frames_a_saltar, max(0, frames_restantes - reserva_final))
                video_finalizado = False
                frames_saltados = 0
                for _ in range(frames_a_saltar):
                    if not captura.grab():
                        video_finalizado = True
                        break
                    frames_saltados += 1
                if frames_saltados:
                    inicio_reproduccion += frames_saltados * duracion_frame_video
                if video_finalizado:
                    break

    captura.release()
    detener_inferencia.set()
    evento_inferencia.set()
    if hilo_inferencia and hilo_inferencia.is_alive():
        hilo_inferencia.join(timeout=1.0)
    if mostrar:
        cv2.destroyAllWindows()
    print(f"[OK] Inferencia finalizada. Frames procesados: {frame_numero}")
    print(f"[OK] Logs: {carpeta_registros / 'predicciones.csv'}")


def revisar_estado(configuracion: Dict[str, object]) -> None:
    """Muestra un resumen rápido de carpetas, muestras y modelos disponibles."""
    rutas = configuracion["rutas"]
    rostros = listar_imagenes_por_identidad(rutas["rostros"], "rostro")
    reid = listar_imagenes_por_identidad(rutas["reidentificacionF"], "reidentificacionF")
    modelos = sorted(Path(rutas["modelos"]).glob("*.pkl"))

    print("\n===== ESTADO DEL PROYECTO =====")
    print(f"Rostros guardados: {len(rostros)}")
    print(f"Muestras Re-ID F guardadas: {len(reid)}")
    conteo_rostros: Dict[str, int] = {}
    conteo_reid: Dict[str, int] = {}
    for muestra in rostros:
        conteo_rostros[muestra.identidad] = conteo_rostros.get(muestra.identidad, 0) + 1
    for muestra in reid:
        conteo_reid[muestra.identidad] = conteo_reid.get(muestra.identidad, 0) + 1
    if conteo_rostros:
        print("Rostros por persona:", ", ".join(f"{k}:{v}" for k, v in sorted(conteo_rostros.items())))
    if conteo_reid:
        print("Re-ID por persona:", ", ".join(f"{k}:{v}" for k, v in sorted(conteo_reid.items())))
    max_muestras = int(configuracion.get("entrenamiento", {}).get("max_muestras_por_clase", 0))
    print(f"Max imagenes/clase entrenamiento: {'todas' if max_muestras <= 0 else max_muestras}")
    print("Modelos SVM:", ", ".join(m.name for m in modelos) if modelos else "ninguno")
    print("SVM rostro fijo:", "existe" if (Path(rutas["modelos"]) / "svm_rostro.pkl").exists() else "no existe")
    print("SVM Re-ID formal:", "existe" if (Path(rutas["modelos"]) / "svm_reidentificacion.pkl").exists() else "no existe")
    modelo_reid_vivo = str(configuracion.get("aprendizaje_reid_en_vivo", {}).get("modelo_salida", "svm_reidentificacion_en_vivo.pkl"))
    print("SVM Re-ID en vivo:", "existe" if (Path(rutas["modelos"]) / modelo_reid_vivo).exists() else "no existe")
    buffer_reid = Path(rutas["modelos"]) / "buffer_reid_hsv_en_vivo.npz"
    print("Buffer Re-ID en vivo:", "existe" if buffer_reid.exists() else "no existe")
    print("Rutas principales:")
    for clave in ["rostros", "reidentificacionF", "modelos", "registros", "salidas"]:
        print(f"- {clave}: {rutas[clave]}")


def leer_texto_menu(mensaje: str, defecto: Optional[str] = None) -> str:
    """Lee texto de consola con valor por defecto."""
    sufijo = f" [{defecto}]" if defecto not in {None, ""} else ""
    valor = input(f"{mensaje}{sufijo}: ").strip()
    if not valor and defecto is not None:
        return str(defecto)
    return valor


def leer_entero_menu(mensaje: str, defecto: int) -> int:
    """Lee un entero de consola sin cerrar el menu si el usuario se equivoca."""
    valor = input(f"{mensaje} [{defecto}]: ").strip()
    if not valor:
        return defecto
    try:
        return int(valor)
    except ValueError:
        print(f"[AVISO] Valor invalido. Se usara {defecto}.")
        return defecto


def pausar_menu() -> None:
    """Pausa breve para que el usuario lea la salida antes de volver al menu."""
    input("\nPresiona Enter para volver al menu...")


def ejecutar_registro_desde_menu(configuracion: dict, modo_captura: str) -> None:
    """Ejecuta registro de rostro, Re-ID o ambos desde el menu interactivo."""
    identidad = leer_texto_menu("Identidad/persona")
    if not identidad:
        print("[AVISO] Debes escribir una identidad.")
        return

    fuente = leer_texto_menu("Camara, URL o video", obtener_fuente_defecto(configuracion))
    muestras = leer_entero_menu("Cantidad de muestras", 40)
    intervalo = leer_entero_menu("Guardar cada N frames", 5)
    resumen = capturar_muestras_tiempo_real(
        configuracion,
        identidad=identidad,
        fuente=fuente,
        modo_captura=modo_captura,
        muestras_objetivo=muestras,
        intervalo_frames=intervalo,
        mostrar_ventana=bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True)),
    )
    print(f"[OK] Registro completado: {resumen}")


def ejecutar_menu_consola(configuracion: dict) -> None:
    """Menu basico para usar el sistema sin recordar comandos."""
    while True:
        print("\n===== MENU RE-ID =====")
        print("1. Inferir con 4 camaras")
        print("2. Inferir con una camara")
        print("3. Cargar imagen")
        print("4. Cargar video")
        print("5. Entrenar modelos")
        print("6. Ver estado / cuantas imagenes hay")
        print("7. Diagnostico rostro")
        print("8. Registrar rostro")
        print("9. Registrar Re-ID cuerpo/ropa")
        print("10. Registrar rostro + Re-ID")
        print("11. Actualizar carpeta reportes")
        print("12. Capturar ROI rostros desde video")
        print("0. Salir")

        opcion = input("Opcion: ").strip()
        try:
            if opcion == "1":
                multi = configuracion.get("multicamara", {})
                max_camaras = int(multi.get("max_camaras", 4))
                fuentes = obtener_fuentes_multicamara(configuracion, max_camaras=max_camaras)
                ejecutar_inferencia_multicamara(configuracion, fuentes)
            elif opcion == "2":
                fuente = leer_texto_menu("Camara, URL o video", obtener_fuente_defecto(configuracion))
                ejecutar_inferencia(configuracion, fuente)
            elif opcion == "3":
                ruta = leer_texto_menu("Ruta de la imagen")
                if ruta:
                    ejecutar_inferencia(configuracion, ruta)
            elif opcion == "4":
                ruta = leer_texto_menu("Ruta del video")
                if ruta:
                    ejecutar_inferencia(configuracion, ruta)
            elif opcion == "5":
                ejecutar_entrenamiento(configuracion)
                pausar_menu()
            elif opcion == "6":
                revisar_estado(configuracion)
                pausar_menu()
            elif opcion == "7":
                ejecutar_diagnostico(configuracion)
                pausar_menu()
            elif opcion == "8":
                ejecutar_registro_desde_menu(configuracion, "rostro")
            elif opcion == "9":
                ejecutar_registro_desde_menu(configuracion, "reid")
            elif opcion == "10":
                ejecutar_registro_desde_menu(configuracion, "ambos")
            elif opcion == "11":
                actualizar_reportes(configuracion)
                pausar_menu()
            elif opcion == "12":
                ruta = leer_texto_menu("Ruta del video")
                if ruta:
                    intervalo = leer_entero_menu("Guardar cada N frames", 5)
                    max_por_identidad = leer_entero_menu("Max capturas por ID temporal (0 sin limite)", 50)
                    capturar_roi_rostros_desde_video(
                        configuracion,
                        ruta,
                        intervalo_frames=intervalo,
                        max_por_identidad=max_por_identidad,
                        mostrar_ventana=bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True)),
                    )
                    pausar_menu()
            elif opcion == "0":
                print("[OK] Saliendo.")
                return
            else:
                print("[AVISO] Opcion no valida.")
        except KeyboardInterrupt:
            print("\n[OK] Operacion cancelada.")
        except Exception as exc:
            print(f"[AVISO] No se pudo completar la opcion: {exc}")
            pausar_menu()


def ejecutar_menu(configuracion: dict) -> None:
    """Menu grafico basico en Tkinter para seleccionar acciones del sistema."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
    except Exception as exc:
        print(f"[AVISO] No se pudo abrir Tkinter ({exc}). Usando menu de consola.")
        ejecutar_menu_consola(configuracion)
        return

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"[AVISO] No se pudo crear la ventana Tkinter ({exc}). Usando menu de consola.")
        ejecutar_menu_consola(configuracion)
        return
    root.title("Menu Re-ID HOG/HSV/SVM")
    root.geometry("760x720")
    root.minsize(680, 620)

    color_fondo = "#f3f4f6"
    color_panel = "#ffffff"
    color_boton = "#1f2937"
    color_boton_sec = "#374151"
    root.configure(bg=color_fondo)

    contenedor = tk.Frame(root, bg=color_fondo, padx=22, pady=18)
    contenedor.pack(fill="both", expand=True)

    titulo = tk.Label(
        contenedor,
        text="Sistema Re-ID",
        font=("Segoe UI", 20, "bold"),
        bg=color_fondo,
        fg="#111827",
    )
    titulo.pack(anchor="w")

    subtitulo = tk.Label(
        contenedor,
        text="Selecciona una opcion. En camaras o video cierra con la tecla q.",
        font=("Segoe UI", 10),
        bg=color_fondo,
        fg="#4b5563",
    )
    subtitulo.pack(anchor="w", pady=(2, 14))

    cuerpo = tk.Frame(contenedor, bg=color_fondo)
    cuerpo.pack(fill="both", expand=True)

    panel_botones = tk.Frame(cuerpo, bg=color_panel, padx=14, pady=14, relief="solid", bd=1)
    panel_botones.pack(side="left", fill="y")

    panel_salida = tk.Frame(cuerpo, bg=color_panel, padx=14, pady=14, relief="solid", bd=1)
    panel_salida.pack(side="right", fill="both", expand=True, padx=(16, 0))

    salida_titulo = tk.Label(panel_salida, text="Salida", font=("Segoe UI", 12, "bold"), bg=color_panel, fg="#111827")
    salida_titulo.pack(anchor="w")

    salida = tk.Text(panel_salida, height=22, wrap="word", font=("Consolas", 9), bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb")
    salida.pack(fill="both", expand=True, pady=(8, 0))
    salida.insert("1.0", "Listo. Elige una opcion del menu.\n")
    salida.configure(state="disabled")

    def escribir_salida(texto: str) -> None:
        salida.configure(state="normal")
        salida.delete("1.0", "end")
        salida.insert("1.0", texto or "[OK] Operacion completada.")
        salida.configure(state="disabled")

    def capturar_salida(funcion, *args) -> str:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            funcion(*args)
        return buffer.getvalue()

    def ejecutar_con_opencv(funcion, *args) -> None:
        root.withdraw()
        try:
            funcion(*args)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
        finally:
            root.deiconify()
            root.lift()

    def inferir_4_camaras() -> None:
        multi = configuracion.get("multicamara", {})
        max_camaras = int(multi.get("max_camaras", 4))
        fuentes = obtener_fuentes_multicamara(configuracion, max_camaras=max_camaras)
        ejecutar_con_opencv(ejecutar_inferencia_multicamara, configuracion, fuentes)

    def inferir_una_camara() -> None:
        fuente = simpledialog.askstring("Camara individual", "Camara, URL o ruta:", initialvalue=obtener_fuente_defecto(configuracion), parent=root)
        if fuente:
            ejecutar_con_opencv(ejecutar_inferencia, configuracion, fuente)

    def cargar_imagen_menu() -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if ruta:
            texto = capturar_salida(ejecutar_inferencia, configuracion, ruta)
            escribir_salida(texto)
            messagebox.showinfo("Imagen procesada", texto or "Imagen procesada.")

    def cargar_video_menu() -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")],
        )
        if ruta:
            ejecutar_con_opencv(ejecutar_inferencia, configuracion, ruta)

    def entrenar_menu() -> None:
        try:
            escribir_salida(capturar_salida(ejecutar_entrenamiento, configuracion))
            messagebox.showinfo("Entrenamiento", "Entrenamiento finalizado.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def estado_menu() -> None:
        escribir_salida(capturar_salida(revisar_estado, configuracion))

    def diagnostico_menu() -> None:
        try:
            escribir_salida(capturar_salida(ejecutar_diagnostico, configuracion))
            messagebox.showinfo("Diagnostico", "Diagnostico generado.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def actualizar_reportes_menu() -> None:
        try:
            escribir_salida(capturar_salida(actualizar_reportes, configuracion))
            messagebox.showinfo("Reportes", "Carpeta reportes actualizada.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def capturar_rostros_video_menu() -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        intervalo = simpledialog.askinteger(
            "Capturar rostros",
            "Guardar cada N frames:",
            initialvalue=5,
            minvalue=1,
            parent=root,
        )
        if not intervalo:
            return
        max_por_identidad = simpledialog.askinteger(
            "Capturar rostros",
            "Max capturas por ID temporal (0 sin limite):",
            initialvalue=50,
            minvalue=0,
            parent=root,
        )
        if max_por_identidad is None:
            return
        ejecutar_con_opencv(
            capturar_roi_rostros_desde_video,
            configuracion,
            ruta,
            intervalo,
            max_por_identidad,
            bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True)),
        )

    def registrar_menu(modo_captura: str) -> None:
        identidad = simpledialog.askstring("Registro", "Identidad/persona:", parent=root)
        if not identidad:
            return
        fuente = simpledialog.askstring("Registro", "Camara, URL o video:", initialvalue=obtener_fuente_defecto(configuracion), parent=root)
        if not fuente:
            return
        muestras = simpledialog.askinteger("Registro", "Cantidad de muestras:", initialvalue=40, minvalue=1, parent=root)
        if not muestras:
            return
        intervalo = simpledialog.askinteger("Registro", "Guardar cada N frames:", initialvalue=5, minvalue=1, parent=root)
        if not intervalo:
            return
        ejecutar_con_opencv(
            capturar_muestras_tiempo_real,
            configuracion,
            identidad,
            fuente,
            modo_captura,
            muestras,
            intervalo,
            bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True)),
        )

    def boton(texto: str, comando, color: str = color_boton) -> None:
        tk.Button(
            panel_botones,
            text=texto,
            command=comando,
            width=28,
            anchor="w",
            padx=12,
            pady=8,
            bg=color,
            fg="white",
            activebackground="#111827",
            activeforeground="white",
            font=("Segoe UI", 10),
            relief="flat",
        ).pack(fill="x", pady=4)

    boton("1. Inferir con 4 camaras", inferir_4_camaras)
    boton("2. Inferir con una camara", inferir_una_camara)
    boton("3. Cargar imagen", cargar_imagen_menu)
    boton("4. Cargar video", cargar_video_menu)
    boton("5. Entrenar modelos", entrenar_menu, color_boton_sec)
    boton("6. Ver estado / imagenes", estado_menu, color_boton_sec)
    boton("7. Diagnostico rostro", diagnostico_menu, color_boton_sec)
    boton("8. Registrar rostro", lambda: registrar_menu("rostro"), color_boton_sec)
    boton("9. Registrar Re-ID cuerpo/ropa", lambda: registrar_menu("reid"), color_boton_sec)
    boton("10. Registrar rostro + Re-ID", lambda: registrar_menu("ambos"), color_boton_sec)
    boton("11. Actualizar reportes", actualizar_reportes_menu, color_boton_sec)
    boton("12. Capturar ROI rostros desde video", capturar_rostros_video_menu, color_boton_sec)

    tk.Button(
        panel_botones,
        text="Salir",
        command=root.destroy,
        width=28,
        anchor="w",
        padx=12,
        pady=8,
        bg="#b91c1c",
        fg="white",
        activebackground="#7f1d1d",
        activeforeground="white",
        font=("Segoe UI", 10),
        relief="flat",
    ).pack(fill="x", pady=(16, 4))

    root.mainloop()


def main() -> None:
    """Punto de entrada del proyecto."""
    argumentos = crear_parser().parse_args()
    configuracion = cargar_configuracion(argumentos.config)
    if argumentos.max_img_entrenamiento is not None:
        configuracion.setdefault("entrenamiento", {})["max_muestras_por_clase"] = argumentos.max_img_entrenamiento
    if argumentos.sin_ventana:
        configuracion.setdefault("ejecucion", {})["mostrar_ventana"] = False
    crear_directorios_base(configuracion)

    fuente = argumentos.fuente or obtener_fuente_defecto(configuracion)

    if argumentos.modo == "menu":
        ejecutar_menu(configuracion)
        return

    if argumentos.modo == "demo":
        crear_dataset_demo(configuracion)
        ejecutar_entrenamiento(configuracion)
        revisar_estado(configuracion)
        return

    if argumentos.modo == "revisar":
        revisar_estado(configuracion)
        return

    if argumentos.modo == "diagnostico":
        ejecutar_diagnostico(configuracion)
        return

    if argumentos.modo == "entrenar":
        ejecutar_entrenamiento(configuracion)
        return

    if argumentos.modo == "capturar_rostros_video":
        capturar_roi_rostros_desde_video(
            configuracion,
            fuente,
            intervalo_frames=argumentos.intervalo,
            max_por_identidad=argumentos.muestras,
            mostrar_ventana=not argumentos.sin_ventana,
        )
        return

    if argumentos.modo in {"registrar_reid", "registrar_rostro", "registrar"}:
        if not argumentos.identidad:
            raise ValueError("Para registrar debes indicar --identidad, ejemplo: --identidad John")
        modo_captura = {
            "registrar_reid": "reid",
            "registrar_rostro": "rostro",
            "registrar": "ambos",
        }[argumentos.modo]
        resumen = capturar_muestras_tiempo_real(
            configuracion,
            identidad=argumentos.identidad,
            fuente=fuente,
            modo_captura=modo_captura,
            muestras_objetivo=argumentos.muestras,
            intervalo_frames=argumentos.intervalo,
            mostrar_ventana=not argumentos.sin_ventana,
        )
        print(f"[OK] Registro completado: {resumen}")
        if argumentos.auto_entrenar:
            ejecutar_entrenamiento(configuracion)
        return

    if argumentos.modo == "inferir":
        multi = configuracion.get("multicamara", {})
        inferir_sin_fuente = argumentos.fuente is None
        mostrar = bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True))
        if inferir_sin_fuente and mostrar and bool(multi.get("activo_sin_fuente_en_inferir", True)):
            max_camaras = int(multi.get("max_camaras", 4))
            fuentes = obtener_fuentes_multicamara(configuracion, max_camaras=max_camaras)
            ejecutar_inferencia_multicamara(configuracion, fuentes)
        else:
            ejecutar_inferencia(configuracion, fuente)
        return


if __name__ == "__main__":
    main()
