# Proyecto Re-ID HOG/HSV/SVM - PC Python

Proyecto ajustado a la documentación oficial. La PC ejecuta toda la IA y los celulares solo se usan como cámaras o fuentes de video.

## Regla principal respetada

1. **Primero identificación facial:** si el rostro se ve bien, se extrae **HoG** del rostro y se clasifica con **SVM facial** ya entrenado con `datos/rostros/<identidad>/`.
2. **Después Re-ID:** si el rostro no se ve, está borroso, está de espaldas o el score facial no supera el umbral, se activa **HSV del torso/ropa + SVM Re-ID**.
3. **Re-ID en vivo:** durante la ejecución, cuando una persona sí fue reconocida por rostro, el sistema guarda el HSV de su torso con esa identidad y va reentrenando el **SVM Re-ID** en vivo. Así después puede reconocerla cuando deje de verse el rostro.
4. **No se fuerza identidad:** si no hay score suficiente o falta un SVM entrenado, se marca como `desconocido`.

## Instalación en Windows

```bat
cd Proyecto_ReID_HOG_HSV_Documentacion_OK
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Coloca YOLO en:

```text
modelos/yolov8n.pt
```

También puedes dejar que Ultralytics lo descargue la primera vez si tienes internet:

```bat
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Orden correcto para ejecutar todo

### Menu basico

```bat
python principal.py
```

Abre una ventana basica con botones usando Tkinter. Desde ahi puedes elegir inferencia con 4 camaras, camara individual, imagen, video, entrenar, actualizar reportes, revisar conteos y registrar muestras. Si Tkinter no esta disponible en tu instalacion de Python, cae al menu de consola.

### 1. Revisar estado

```bat
python principal.py --modo revisar
```

### 2. Registrar/capturar rostros para el dataset facial

Esto llena `datos/rostros/<identidad>/`. Deben verse rostros claros.

```bat
python principal.py --modo registrar_rostro --identidad John --fuente 0 --muestras 25
python principal.py --modo registrar_rostro --identidad Matias --fuente 0 --muestras 25
```

### 3. Entrenar el SVM facial HoG

```bat
python principal.py --modo entrenar
```

Aquí se genera:

```text
modelos/svm_rostro.pkl
```

### 4. Ejecutar inferencia en vivo

```bat
python principal.py --modo inferir --fuente 0
```

Durante esta ejecución pasa esto:

- Si ve la cara y reconoce con SVM facial: muestra `rostro_hog_svm`.
- Mientras reconoce por rostro, guarda HSV del torso y entrena Re-ID en vivo.
- Si luego la persona se gira o no se ve la cara: activa `reid_hsv_svm`.
- Si no hay suficiente confianza: muestra `desconocido`.

### 5. Usar celular como cámara

Con IP Webcam o DroidCam usa la URL de video:

```bat
python principal.py --modo inferir --fuente http://192.168.1.20:8080/video
```

## Registro Re-ID controlado opcional

Este modo es opcional. Sirve para llenar `datos/reidentificacion/<identidad>/` con torso/ropa desde cámara y entrenar SVM Re-ID con datos controlados.

```bat
python principal.py --modo registrar_reid --identidad John --fuente 0 --muestras 40
python principal.py --modo registrar_reid --identidad Matias --fuente 0 --muestras 40
python principal.py --modo entrenar
```

## Demo técnica sin cámara

Solo valida que el entrenamiento Re-ID funciona con datos sintéticos. No representa personas reales.

```bat
python principal.py --modo demo --sin-ventana
```

## Carpetas principales

```text
datos/rostros/                 # dataset facial para HoG + SVM facial
datos/reidentificacion/        # torso/ropa para HSV + SVM Re-ID opcional
modelos/                       # svm_rostro.pkl y svm_reidentificacion.pkl
modelos/buffer_reid_hsv_en_vivo.npz # HSV recolectado durante inferencia
registros/                     # logs y metadata CSV
salidas/                       # imágenes procesadas
reportes/                      # diagnostico, matriz de confusion y metricas
```

## Tecla de salida

- `q`: cerrar captura o inferencia.
