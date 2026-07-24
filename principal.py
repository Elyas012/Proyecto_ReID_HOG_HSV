"""Entrada principal del proyecto Re-ID PC Python + cámaras móviles.

Comandos principales:
- registrar_reid: captura torso/ropa desde cámara para entrenar SVM Re-ID.
- registrar_rostro: captura rostros visibles para identificación facial.
- registrar: captura rostro y Re-ID en la misma sesión.
- entrenar: entrena SVM facial y SVM Re-ID con las capturas disponibles.
- inferir: procesa imagen, video, webcam o URL IP/RTSP/HTTP.
- demo: crea datos sintéticos y valida que el pipeline entrene sin cámara.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable

import cv2

from sistema_reid.captura_tiempo_real import (
    capturar_muestras_tiempo_real,
    crear_dataset_demo,
    entrenar_desde_capturas,
)
from sistema_reid.configuracion import cargar_configuracion, crear_directorios_base
from sistema_reid.datos import cargar_imagen_bgr, listar_imagenes_por_identidad
from sistema_reid.evaluacion import diagnosticar_modelo_rostro
from sistema_reid.inferencia import MotorInferencia, ResultadoIdentidad, dibujar_resultados
from sistema_reid.panel_control import PanelControlInferencia


def crear_parser() -> argparse.ArgumentParser:
    """Crea los argumentos de consola del sistema."""
    parser = argparse.ArgumentParser(description="Sistema Re-ID HoG/HSV/SVM para PC con cámaras móviles")
    parser.add_argument("--config", default="configuracion.yaml", help="Ruta del archivo de configuración")
    parser.add_argument(
        "--modo",
        choices=["registrar_reid", "registrar_rostro", "registrar", "entrenar", "inferir", "demo", "revisar", "diagnostico"],
        required=True,
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


def ejecutar_entrenamiento(configuracion: dict) -> None:
    """Entrena SVM facial HoG y SVM Re-ID HSV con las capturas actuales."""
    modelos = entrenar_desde_capturas(configuracion)
    if modelos:
        print(f"[OK] Artefactos generados: {', '.join(modelos.keys())}")
    else:
        print("[AVISO] No se entrenó ningún modelo. Registra rostros con --modo registrar_rostro y/o torso con --modo registrar_reid.")


def ejecutar_diagnostico(configuracion: dict) -> None:
    """Genera metricas del SVM facial y matriz de confusion."""
    metricas = diagnosticar_modelo_rostro(configuracion)
    carpeta_reportes = Path(configuracion["rutas"]["reportes"])
    print("[OK] Diagnostico facial generado")
    print(f"[OK] Accuracy: {metricas.get('accuracy', 0):.4f} | F1 macro: {metricas.get('f1_macro', 0):.4f}")
    print(f"[OK] Reporte: {carpeta_reportes / 'diagnostico_rostro.txt'}")
    print(f"[OK] Matriz CSV: {carpeta_reportes / 'matriz_confusion_rostro.csv'}")
    print(f"[OK] Matriz imagen: {carpeta_reportes / 'matriz_confusion_rostro.jpg'}")


def abrir_fuente(fuente: str):
    """Abre una imagen, video, URL o cámara según el valor recibido."""
    fuente = str(fuente)
    if fuente.isdigit() or fuente.startswith(("http://", "https://", "rtsp://")):
        # Comentario clave: aquí entran webcams locales, celulares por IP Webcam/DroidCam y RTSP.
        return cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente), "video"

    ruta = Path(fuente)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la fuente: {fuente}")

    if ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        imagen = cargar_imagen_bgr(ruta)
        return imagen, "imagen"
    return cv2.VideoCapture(str(ruta)), "video"


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


def ejecutar_inferencia(configuracion: dict, fuente: str) -> None:
    """Ejecuta inferencia sobre imagen, video, URL o cámara."""
    motor = MotorInferencia(configuracion)
    motor.cargar_modelos()
    entrada, tipo = abrir_fuente(fuente)
    carpeta_salida = Path(configuracion["rutas"]["salidas"])
    carpeta_registros = Path(configuracion["rutas"]["registros"])
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    carpeta_registros.mkdir(parents=True, exist_ok=True)

    if tipo == "imagen":
        resultados = motor.procesar_frame(entrada)
        salida = dibujar_resultados(entrada, resultados)
        ruta_salida = carpeta_salida / "resultado_imagen.jpg"
        cv2.imwrite(str(ruta_salida), salida)
        guardar_log_predicciones(carpeta_registros / "predicciones.csv", 1, resultados, 0.0)
        print(f"[OK] Detecciones procesadas: {len(resultados)}")
        print(f"[OK] Resultado guardado en: {ruta_salida}")
        return

    captura = entrada
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    frame_numero = 0
    ultimo_tiempo = time.time()
    mostrar = bool(configuracion.get("ejecucion", {}).get("mostrar_ventana", True))
    usar_panel = bool(configuracion.get("panel_control", {}).get("activo", True))
    nombre_ventana = "Re-ID PC Python"
    panel = PanelControlInferencia(nombre_ventana, configuracion) if mostrar and usar_panel else None
    if mostrar and panel is None:
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = captura.read()
        if not ok:
            break
        frame_numero += 1

        ahora = time.time()
        fps = 1.0 / max(0.001, ahora - ultimo_tiempo)
        ultimo_tiempo = ahora

        resultados = motor.procesar_frame(frame)
        salida = dibujar_resultados(frame, resultados)
        guardar_log_predicciones(carpeta_registros / "predicciones.csv", frame_numero, resultados, fps)

        # Comentario clave: la ventana permite validar visualmente el prototipo durante la exposición.
        if mostrar:
            vista = panel.construir_vista(salida, resultados, fps, frame_numero) if panel else salida
            cv2.imshow(nombre_ventana, vista)
            tecla = cv2.waitKey(1) & 0xFF
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

    captura.release()
    if mostrar:
        cv2.destroyAllWindows()
    print(f"[OK] Inferencia finalizada. Frames procesados: {frame_numero}")
    print(f"[OK] Logs: {carpeta_registros / 'predicciones.csv'}")


def revisar_estado(configuracion: Dict[str, object]) -> None:
    """Muestra un resumen rápido de carpetas, muestras y modelos disponibles."""
    rutas = configuracion["rutas"]
    rostros = listar_imagenes_por_identidad(rutas["rostros"], "rostro")
    reid = listar_imagenes_por_identidad(rutas["reidentificacion"], "reidentificacion")
    modelos = sorted(Path(rutas["modelos"]).glob("*.pkl"))

    print("\n===== ESTADO DEL PROYECTO =====")
    print(f"Rostros guardados: {len(rostros)}")
    print(f"Muestras Re-ID guardadas: {len(reid)}")
    max_muestras = int(configuracion.get("entrenamiento", {}).get("max_muestras_por_clase", 0))
    print(f"Max imagenes/clase entrenamiento: {'todas' if max_muestras <= 0 else max_muestras}")
    print("Modelos SVM:", ", ".join(m.name for m in modelos) if modelos else "ninguno")
    print("SVM rostro fijo:", "existe" if (Path(rutas["modelos"]) / "svm_rostro.pkl").exists() else "no existe")
    print("SVM Re-ID fijo:", "existe" if (Path(rutas["modelos"]) / "svm_reidentificacion.pkl").exists() else "no existe")
    modelo_reid_vivo = str(configuracion.get("aprendizaje_reid_en_vivo", {}).get("modelo_salida", "svm_reidentificacion_en_vivo.pkl"))
    print("SVM Re-ID en vivo:", "existe" if (Path(rutas["modelos"]) / modelo_reid_vivo).exists() else "no existe")
    buffer_reid = Path(rutas["modelos"]) / "buffer_reid_hsv_en_vivo.npz"
    print("Buffer Re-ID en vivo:", "existe" if buffer_reid.exists() else "no existe")
    print("Rutas principales:")
    for clave in ["rostros", "reidentificacion", "modelos", "registros", "salidas"]:
        print(f"- {clave}: {rutas[clave]}")


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
        ejecutar_inferencia(configuracion, fuente)
        return


if __name__ == "__main__":
    main()
