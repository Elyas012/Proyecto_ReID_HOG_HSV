"""Lectura y preparación de la configuración del proyecto."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIGURACION_DEFECTO: Dict[str, Any] = {
    "proyecto": {"nombre": "Proyecto_ReID_HOG_HSV_Documentacion_OK"},
    "ejecucion": {"dispositivo": "cpu", "mostrar_ventana": True, "guardar_salida": True},
    "rendimiento": {"reservar_nucleos": 1, "hilos_cpu": 0, "hilos_cpu_multicamara": 0},
    "multicamara": {
        "activo_sin_fuente_en_inferir": True,
        "max_camaras": 4,
        "ancho_proceso": 640,
        "ancho_celda": 640,
        "alto_celda": 360,
        "intervalo_inferencia_ms": 120,
        "tamano_yolo": 416,
    },
    "video": {
        "ancho_proceso": 960,
        "ancho_vista": 960,
        "alto_vista": 540,
        "procesar_cada_n_frames": 2,
        "inferencia_async": True,
        "respetar_fps": True,
        "saltar_frames_atrasados": True,
        "max_salto_frames": 30,
        "fps_por_defecto": 30,
        "tamano_yolo": 416,
    },
    "yolo": {"pesos": "modelos/yolov8n.pt", "clase_persona": 0, "confianza": 0.40, "tamano_imagen": 640},
    "umbrales": {"score_rostro": 0.80, "margen_rostro": 0.12, "score_reid": 0.65, "nitidez_minima": 60.0, "tamano_minimo_rostro": 40},
    "caracteristicas": {"tamano_rostro": [96, 96], "tamano_torso": [128, 256], "bins_hsv": [16, 16, 8]},
    "rutas": {
        "datos": "datos",
        "rostros": "datos/rostros",
        "reidentificacion": "datos/reidentificacion",
        "pruebas_rostros": "datos/pruebas/rostros",
        "pruebas_reidentificacion": "datos/pruebas/reidentificacion",
        "desconocidos": "datos/desconocidos",
        "modelos": "modelos",
        "salidas": "salidas",
        "reportes": "reportes",
        "registros": "registros",
    },
    "entrenamiento": {"kernel": "rbf", "validacion": 0.20, "semilla": 42, "probabilidad": True, "max_muestras_por_clase": 0},
    "aprendizaje_reid_en_vivo": {
        "activo": True,
        "minimo_por_identidad": 4,
        "reentrenar_cada": 8,
        "modelo_salida": "svm_reidentificacion_en_vivo.pkl",
        "usar_modelo_en_vivo": True,
    },
}


def unir_diccionarios(base: Dict[str, Any], nuevo: Dict[str, Any]) -> Dict[str, Any]:
    """Une dos diccionarios anidados sin perder valores por defecto."""
    resultado = deepcopy(base)
    for clave, valor in nuevo.items():
        # Comentario clave: si ambos valores son diccionarios, se fusionan por niveles.
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = unir_diccionarios(resultado[clave], valor)
        else:
            resultado[clave] = valor
    return resultado


def cargar_configuracion(ruta_configuracion: str | Path = "configuracion.yaml") -> Dict[str, Any]:
    """Carga configuracion.yaml y completa cualquier dato faltante con valores seguros."""
    ruta = Path(ruta_configuracion)
    datos_archivo: Dict[str, Any] = {}

    # Comentario clave: si no existe el YAML, el sistema todavía puede arrancar con valores base.
    if ruta.exists():
        with ruta.open("r", encoding="utf-8") as archivo:
            datos_archivo = yaml.safe_load(archivo) or {}

    configuracion = unir_diccionarios(CONFIGURACION_DEFECTO, datos_archivo)
    configuracion["_base"] = str(ruta.resolve().parent)
    return resolver_rutas(configuracion)


def resolver_rutas(configuracion: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte las rutas relativas del YAML en rutas absolutas listas para usar."""
    base = Path(configuracion.get("_base", ".")).resolve()
    rutas = configuracion.get("rutas", {})

    for nombre, ruta in list(rutas.items()):
        ruta_path = Path(str(ruta))
        # Comentario clave: las rutas absolutas no se modifican; las relativas se resuelven desde el proyecto.
        rutas[nombre] = str(ruta_path if ruta_path.is_absolute() else base / ruta_path)

    pesos = Path(str(configuracion.get("yolo", {}).get("pesos", "modelos/yolov8n.pt")))
    if not pesos.is_absolute():
        configuracion["yolo"]["pesos"] = str(base / pesos)
    return configuracion


def crear_directorios_base(configuracion: Dict[str, Any]) -> None:
    """Crea las carpetas principales para evitar errores al guardar resultados."""
    for ruta in configuracion.get("rutas", {}).values():
        # Comentario clave: exist_ok=True permite ejecutar el script varias veces sin borrar datos.
        Path(str(ruta)).mkdir(parents=True, exist_ok=True)
