"""Carga de datasets organizados por identidad."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .caracteristicas import extraer_histograma_hsv, extraer_hog_rostro, rostro_es_util
from .deteccion import DetectorRostros

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MuestraImagen:
    """Informacion minima de una imagen dentro del dataset."""

    ruta: Path
    identidad: str
    tipo: str


def listar_imagenes_por_identidad(carpeta_base: str | Path, tipo: str) -> List[MuestraImagen]:
    """Lista imagenes ubicadas en subcarpetas por identidad."""
    carpeta = Path(carpeta_base)
    muestras: List[MuestraImagen] = []
    if not carpeta.exists():
        return muestras

    for carpeta_identidad in sorted(carpeta.iterdir()):
        if not carpeta_identidad.is_dir():
            continue
        identidad = carpeta_identidad.name
        for ruta in sorted(carpeta_identidad.rglob("*")):
            if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                muestras.append(MuestraImagen(ruta=ruta, identidad=identidad, tipo=tipo))
    return muestras


def es_imagen_augmentation_reid(ruta: str | Path) -> bool:
    """Identifica copias aumentadas Re-ID por nombre de archivo."""
    return "_aug_" in Path(ruta).stem.lower()


def limitar_imagenes_recientes_por_identidad(carpeta_identidad: str | Path, maximo: int) -> int:
    """Conserva solo las imagenes mas recientes dentro de una carpeta de identidad."""
    carpeta = Path(carpeta_identidad)
    if maximo <= 0 or not carpeta.exists():
        return 0

    imagenes = [ruta for ruta in carpeta.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN]
    if len(imagenes) <= maximo:
        return 0

    imagenes.sort(key=lambda ruta: (ruta.stat().st_mtime, ruta.name), reverse=True)
    eliminadas = 0
    for ruta in imagenes[maximo:]:
        try:
            ruta.unlink()
            eliminadas += 1
        except OSError as exc:
            print(f"[AVISO] No se pudo eliminar imagen antigua Re-ID: {ruta} ({exc})")
    return eliminadas


def limitar_muestras_por_identidad(
    muestras: Sequence[MuestraImagen],
    maximo_por_identidad: Optional[int],
    semilla: int = 42,
) -> List[MuestraImagen]:
    """Limita fotos por identidad antes de extraer descriptores costosos."""
    if maximo_por_identidad is None or maximo_por_identidad <= 0:
        return list(muestras)

    rng = np.random.default_rng(semilla)
    seleccionadas: List[MuestraImagen] = []
    identidades = sorted({muestra.identidad for muestra in muestras})

    for identidad in identidades:
        grupo = [muestra for muestra in muestras if muestra.identidad == identidad]
        if len(grupo) > maximo_por_identidad:
            indices = sorted(rng.choice(len(grupo), size=maximo_por_identidad, replace=False).tolist())
            grupo = [grupo[indice] for indice in indices]
        seleccionadas.extend(grupo)

    return seleccionadas


def filtrar_muestras_minimas_por_identidad(
    muestras: Sequence[MuestraImagen],
    minimo_por_identidad: Optional[int],
) -> List[MuestraImagen]:
    """Omite identidades que no alcanzan el minimo configurado de muestras."""
    if minimo_por_identidad is None or minimo_por_identidad <= 0:
        return list(muestras)

    conteo: Dict[str, int] = {}
    for muestra in muestras:
        conteo[muestra.identidad] = conteo.get(muestra.identidad, 0) + 1

    omitidas = {identidad: total for identidad, total in conteo.items() if total < minimo_por_identidad}
    if omitidas:
        print(f"[AVISO] Identidades Re-ID omitidas por minimo de muestras: {omitidas}")

    return [muestra for muestra in muestras if conteo.get(muestra.identidad, 0) >= minimo_por_identidad]


def cargar_imagen_bgr(ruta: str | Path) -> np.ndarray:
    """Lee una imagen desde disco en formato BGR, incluso con rutas Unicode en Windows."""
    ruta = Path(ruta)
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return imagen


def guardar_imagen_bgr(ruta: str | Path, imagen: np.ndarray) -> None:
    """Guarda una imagen BGR soportando rutas Unicode en Windows."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    extension = ruta.suffix or ".jpg"
    ok, codificada = cv2.imencode(extension, imagen)
    if not ok:
        raise RuntimeError(f"No se pudo codificar la imagen: {ruta}")
    codificada.tofile(str(ruta))


def _float_config(configuracion: Dict[str, object], clave: str, defecto: float, minimo: float, maximo: float) -> float:
    """Lee un float de configuracion y lo mantiene en un rango seguro."""
    try:
        valor = float(configuracion.get(clave, defecto))
    except (TypeError, ValueError):
        valor = defecto
    return max(minimo, min(maximo, valor))


def _int_config(configuracion: Dict[str, object], clave: str, defecto: int, minimo: int, maximo: int) -> int:
    """Lee un entero de configuracion y lo mantiene en un rango seguro."""
    try:
        valor = int(configuracion.get(clave, defecto))
    except (TypeError, ValueError):
        valor = defecto
    return max(minimo, min(maximo, valor))


def aplicar_augmentation_reid(imagen: np.ndarray, rng: np.random.Generator, configuracion: Dict[str, object]) -> np.ndarray:
    """Crea una variacion suave para Re-ID sin cambiar la identidad visual de la ropa."""
    salida = imagen.copy()
    alto, ancho = salida.shape[:2]

    escala_min = _float_config(configuracion, "recorte_escala_min", 0.92, 0.70, 1.0)
    if escala_min < 1.0 and alto > 20 and ancho > 20:
        escala = float(rng.uniform(escala_min, 1.0))
        nuevo_ancho = max(1, int(ancho * escala))
        nuevo_alto = max(1, int(alto * escala))
        x1 = int(rng.integers(0, max(1, ancho - nuevo_ancho + 1)))
        y1 = int(rng.integers(0, max(1, alto - nuevo_alto + 1)))
        salida = salida[y1 : y1 + nuevo_alto, x1 : x1 + nuevo_ancho]
        salida = cv2.resize(salida, (ancho, alto), interpolation=cv2.INTER_AREA)

    rotacion = _float_config(configuracion, "rotacion_grados", 3.0, 0.0, 15.0)
    if rotacion > 0:
        angulo = float(rng.uniform(-rotacion, rotacion))
        matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1.0)
        salida = cv2.warpAffine(salida, matriz, (ancho, alto), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    rango_contraste = _float_config(configuracion, "contraste", 0.12, 0.0, 0.50)
    rango_brillo = _float_config(configuracion, "brillo", 18.0, 0.0, 80.0)
    contraste = 1.0 + float(rng.uniform(-rango_contraste, rango_contraste))
    brillo = float(rng.uniform(-rango_brillo, rango_brillo))
    salida = np.clip(salida.astype("float32") * contraste + brillo, 0, 255).astype("uint8")

    ruido_std = _float_config(configuracion, "ruido_std", 3.0, 0.0, 25.0)
    if ruido_std > 0:
        ruido = rng.normal(0.0, ruido_std, salida.shape).astype("float32")
        salida = np.clip(salida.astype("float32") + ruido, 0, 255).astype("uint8")

    if bool(configuracion.get("espejo_horizontal", True)) and bool(rng.integers(0, 2)):
        salida = cv2.flip(salida, 1)

    return salida


def obtener_roi_rostro_entrenamiento(imagen: np.ndarray, detector_rostros: DetectorRostros) -> np.ndarray | None:
    """Devuelve solo un ROI de rostro para entrenar identificacion facial."""
    rostros = detector_rostros.detectar_rostros(imagen, tamano_minimo=40)
    if rostros:
        return rostros[0].roi
    return None


def construir_dataset_rostros(
    carpeta_rostros: str | Path,
    max_muestras_por_clase: Optional[int] = None,
    semilla: int = 42,
    omitir_sin_roi: bool = True,
    filtrar_baja_calidad: bool = True,
    tamano_minimo: int = 40,
    nitidez_minima: float = 60.0,
    parametros_hog: Optional[Dict[str, object]] = None,
    configuracion: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HoG y etiquetas desde datos/rostros/<identidad>."""
    muestras = listar_imagenes_por_identidad(carpeta_rostros, tipo="rostro")
    muestras = limitar_muestras_por_identidad(muestras, max_muestras_por_clase, semilla)
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    muestras_usadas: List[MuestraImagen] = []
    omitidas = 0
    usadas_sin_roi = 0
    detector_rostros = DetectorRostros.desde_config(configuracion or {})
    omitidas_calidad = 0
    parametros_hog = parametros_hog or {}

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        roi_rostro = obtener_roi_rostro_entrenamiento(imagen, detector_rostros)
        if roi_rostro is None:
            if omitir_sin_roi:
                omitidas += 1
                continue
            roi_rostro = imagen
            usadas_sin_roi += 1
        elif filtrar_baja_calidad and not rostro_es_util(roi_rostro, tamano_minimo=tamano_minimo, nitidez_minima=nitidez_minima):
            omitidas_calidad += 1
            continue
        vectores.append(extraer_hog_rostro(roi_rostro, **parametros_hog))
        etiquetas.append(muestra.identidad)
        muestras_usadas.append(muestra)

    if omitidas:
        print(f"[AVISO] Rostros omitidos por no detectar ROI facial: {omitidas}")
    if usadas_sin_roi:
        print(f"[AVISO] Rostros sin ROI usados con imagen completa: {usadas_sin_roi}")
    if omitidas_calidad:
        print(f"[AVISO] Rostros omitidos por baja calidad/tamano/nitidez: {omitidas_calidad}")
    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras_usadas


def construir_dataset_reid(
    carpeta_reid: str | Path,
    max_muestras_por_clase: Optional[int] = None,
    min_muestras_por_clase: Optional[int] = None,
    semilla: int = 42,
    parametros_hsv: Optional[Dict[str, object]] = None,
    augmentacion: Optional[Dict[str, object]] = None,
    tipo: str = "reidentificacion",
    incluir_aumentadas: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Construye vectores HSV y etiquetas desde una carpeta Re-ID organizada por identidad."""
    muestras = listar_imagenes_por_identidad(carpeta_reid, tipo=tipo)
    if not incluir_aumentadas:
        muestras = [muestra for muestra in muestras if not es_imagen_augmentation_reid(muestra.ruta)]
    muestras = filtrar_muestras_minimas_por_identidad(muestras, min_muestras_por_clase)
    muestras = limitar_muestras_por_identidad(muestras, max_muestras_por_clase, semilla)
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    muestras_usadas: List[MuestraImagen] = []
    parametros_hsv = parametros_hsv or {}
    augmentacion = augmentacion or {}
    usar_augmentacion = bool(augmentacion.get("activo", False))
    cantidad_aug = _int_config(augmentacion, "cantidad_por_imagen", 1, 0, 20) if usar_augmentacion else 0
    guardar_aug = bool(augmentacion.get("guardar_imagenes", True))
    maximo_por_identidad = _int_config(augmentacion, "max_imagenes_por_identidad", 50, 0, 100000)
    rng = np.random.default_rng(semilla)
    carpetas_para_limpiar: set[Path] = set()

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        vectores.append(extraer_histograma_hsv(imagen, **parametros_hsv))
        etiquetas.append(muestra.identidad)
        muestras_usadas.append(muestra)

        for indice_aug in range(cantidad_aug):
            imagen_aug = aplicar_augmentation_reid(imagen, rng, augmentacion)
            ruta_aug = muestra.ruta
            if guardar_aug:
                carpeta_identidad = muestra.ruta.parent
                nombre_aug = f"{muestra.ruta.stem}_aug_{indice_aug + 1:02d}{muestra.ruta.suffix.lower() or '.jpg'}"
                ruta_aug = carpeta_identidad / nombre_aug
                guardar_imagen_bgr(ruta_aug, imagen_aug)
                carpetas_para_limpiar.add(carpeta_identidad)

            vectores.append(extraer_histograma_hsv(imagen_aug, **parametros_hsv))
            etiquetas.append(muestra.identidad)
            muestras_usadas.append(MuestraImagen(ruta=ruta_aug, identidad=muestra.identidad, tipo="reidentificacion_aug"))

    if cantidad_aug:
        total_aug = len(muestras) * cantidad_aug
        print(f"[INFO] Augmentation Re-ID generado: {total_aug} muestras nuevas desde {len(muestras)} imagenes base.")
        for carpeta_identidad in carpetas_para_limpiar:
            limitar_imagenes_recientes_por_identidad(carpeta_identidad, maximo_por_identidad)

    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras_usadas


def construir_augmentation_reid(
    muestras: Sequence[MuestraImagen],
    semilla: int = 42,
    parametros_hsv: Optional[Dict[str, object]] = None,
    augmentacion: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[MuestraImagen]]:
    """Genera vectores HSV aumentados desde una lista de muestras Re-ID ya seleccionadas."""
    augmentacion = augmentacion or {}
    if not bool(augmentacion.get("activo", False)):
        return np.asarray([], dtype="float32"), np.asarray([]), []

    cantidad_aug = _int_config(augmentacion, "cantidad_por_imagen", 1, 0, 20)
    if cantidad_aug <= 0:
        return np.asarray([], dtype="float32"), np.asarray([]), []

    parametros_hsv = parametros_hsv or {}
    guardar_aug = bool(augmentacion.get("guardar_imagenes", True))
    maximo_por_identidad = _int_config(augmentacion, "max_imagenes_por_identidad", 50, 0, 100000)
    rng = np.random.default_rng(semilla)
    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    muestras_aug: List[MuestraImagen] = []
    carpetas_para_limpiar: set[Path] = set()

    for muestra in muestras:
        imagen = cargar_imagen_bgr(muestra.ruta)
        for indice_aug in range(cantidad_aug):
            imagen_aug = aplicar_augmentation_reid(imagen, rng, augmentacion)
            ruta_aug = muestra.ruta
            if guardar_aug:
                carpeta_identidad = muestra.ruta.parent
                nombre_aug = f"{muestra.ruta.stem}_aug_{indice_aug + 1:02d}{muestra.ruta.suffix.lower() or '.jpg'}"
                ruta_aug = carpeta_identidad / nombre_aug
                guardar_imagen_bgr(ruta_aug, imagen_aug)
                carpetas_para_limpiar.add(carpeta_identidad)

            vectores.append(extraer_histograma_hsv(imagen_aug, **parametros_hsv))
            etiquetas.append(muestra.identidad)
            muestras_aug.append(MuestraImagen(ruta=ruta_aug, identidad=muestra.identidad, tipo="reidentificacionF_aumentada"))

    print(f"[INFO] Augmentation Re-ID generado para entrenamiento: {len(muestras_aug)} muestras nuevas desde {len(muestras)} imagenes base.")
    for carpeta_identidad in carpetas_para_limpiar:
        limitar_imagenes_recientes_por_identidad(carpeta_identidad, maximo_por_identidad)
    return np.asarray(vectores, dtype="float32"), np.asarray(etiquetas), muestras_aug


def guardar_metadata_csv(muestras: Sequence[MuestraImagen], ruta_salida: str | Path) -> None:
    """Guarda un CSV simple para mantener trazabilidad de las imagenes usadas."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["ruta", "identidad", "tipo"])
        escritor.writeheader()
        for muestra in muestras:
            escritor.writerow({"ruta": str(muestra.ruta), "identidad": muestra.identidad, "tipo": muestra.tipo})


def validar_dataset_minimo(etiquetas: Sequence[str], minimo_por_clase: int = 2) -> None:
    """Verifica que existan suficientes imagenes por identidad antes de entrenar."""
    conteo = {etiqueta: list(etiquetas).count(etiqueta) for etiqueta in set(etiquetas)}
    clases_insuficientes = [clase for clase, total in conteo.items() if total < minimo_por_clase]

    if len(conteo) < 2:
        raise ValueError("Se necesitan al menos dos identidades para entrenar SVM.")
    if clases_insuficientes:
        raise ValueError(f"Faltan imagenes en estas identidades: {clases_insuficientes}")
