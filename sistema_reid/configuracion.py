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
        "ancho_proceso": 800,
        "ancho_celda": 640,
        "alto_celda": 360,
        "intervalo_inferencia_ms": 120,
        "tamano_yolo": 416,
    },
    "video": {
        "respetar_resolucion_original": True,
        "vista_original_estricta": False,
        "pantalla_completa": True,
        "panel_separado": True,
        "alto_panel": 720,
        "ancho_proceso": 640,
        "ancho_vista": 1280,
        "alto_vista": 720,
        "procesar_cada_n_frames": 3,
        "modo_fluido": True,
        "max_desfase_roi_frames": 0,
        "respetar_fps": True,
        "saltar_frames_atrasados": True,
        "fps_por_defecto": 30,
        "tamano_yolo": 416,
    },
    "yolo": {"pesos": "modelos/yolov8n.pt", "clase_persona": 0, "confianza": 0.40, "tamano_imagen": 640},
    "umbrales": {"score_rostro": 0.80, "margen_rostro": 0.12, "score_reid": 0.65, "nitidez_minima": 60.0, "tamano_minimo_rostro": 40},
    "visualizacion": {"modo_cajas": "ambas"},
    "rostro_en_persona": {
        "usar_zoom_si_no_detecta": True,
        "factor_zoom": 3.0,
        "factores_zoom": [2.0, 3.0, 4.0],
        "porcentaje_superior": 0.45,
        "tamano_minimo_zoom": 18,
        "usar_roi_zoom_para_clasificar": True,
    },
    "caracteristicas": {
        "tamano_rostro": [128, 128],
        "hog_orientaciones": 9,
        "hog_pixeles_por_celda": [8, 8],
        "hog_celdas_por_bloque": [2, 2],
        "tamano_torso": [128, 256],
        "bins_hsv": [16, 16, 8],
        "reid_bandas_horizontales": 3,
        "reid_histograma_global": True,
        "reid_incluir_momentos": True,
        "reid_recorte_lateral": 0.08,
        "reid_recorte_superior": 0.0,
        "reid_recorte_inferior": 0.0,
    },
    "rutas": {
        "datos": "datos",
        "rostros": "datos/rostros",
        "rostros_v2": "datos/rostros_v2",
        "reidentificacionF": "datos/reidentificacionF",
        "modelos": "modelos",
        "salidas": "salidas",
        "reportes": "reportes",
        "registros": "registros",
    },
    "entrenamiento": {
        "kernel": "rbf",
        "kernel_rostro": "rbf",
        "kernel_reid": "rbf",
        "validacion": 0.20,
        "validacion_por_clase": True,
        "semilla": 42,
        "probabilidad": True,
        "max_muestras_por_clase": 0,
        "max_muestras_rostro_por_clase": 0,
        "min_muestras_reid_por_clase": 0,
        "max_muestras_reid_por_clase": 0,
        "omitir_rostros_sin_roi": True,
        "filtrar_rostros_baja_calidad": True,
    },
    "aprendizaje_reid_en_vivo": {
        "activo": True,
        "combinar_con_reid_fijo": True,
        "guardar_capturas": True,
        "carpeta_capturas": "datos/reidentificacionF",
        "minimo_por_identidad": 4,
        "reentrenar_cada": 8,
        "reentrenar_combinado_al_iniciar": True,
        "modelo_salida": "svm_reidentificacion_combinado.pkl",
        "usar_modelo_en_vivo": True,
        "max_muestras_vivas_por_identidad": 80,
        "max_imagenes_carpeta_por_identidad": 50,
        "score_rostro_min_aprendizaje": 0.90,
        "margen_rostro_min_aprendizaje": 0.20,
    },
    "augmentation_reidF": {
        "activo": True,
        "cantidad_por_imagen": 1,
        "guardar_imagenes": True,
        "max_imagenes_por_identidad": 50,
        "brillo": 18.0,
        "contraste": 0.12,
        "ruido_std": 3.0,
        "rotacion_grados": 3.0,
        "recorte_escala_min": 0.92,
        "espejo_horizontal": True,
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
