# Proyecto Re-ID HOG/HSV/SVM - PC Python

Sistema de escritorio en Python para identificacion facial y re-identificacion de personas. La PC ejecuta toda la IA; las camaras, celulares, videos o imagenes solo entregan la fuente visual.

## Idea principal

El sistema trabaja con dos modelos SVM:

- Identificacion facial: rostro visible -> descriptor HoG -> `svm_rostro.pkl`.
- Re-ID: cuerpo completo/ropa -> descriptor HSV espacial -> `svm_reidentificacion.pkl` o modelo combinado en vivo.

La prioridad siempre es:

1. Detectar persona con YOLO.
2. Buscar ROI de rostro dentro de la persona.
3. Si no encuentra rostro, probar zoom en el 45% superior de la persona.
4. Si el rostro es util, clasificar con HoG + SVM facial.
5. Si el rostro no sirve o el score/margen no alcanza, usar Re-ID HSV + SVM.
6. Si tampoco hay confianza suficiente, mostrar `desconocido`.

## Comando principal

```bat
python principal.py
```

Abre el menu principal con Tkinter. Desde ahi se puede usar camara individual, modo 4 camaras, imagen, video, entrenamiento, diagnostico, registro y reportes.

Tambien se puede abrir directamente:

```bat
python principal.py --modo menu
```

## Comandos disponibles

```bat
python principal.py --modo revisar
python principal.py --modo entrenar
python principal.py --modo diagnostico
python principal.py --modo inferir --fuente 0
python principal.py --modo inferir --fuente ruta\video.mp4
python principal.py --modo inferir --fuente ruta\imagen.jpg
python principal.py --modo registrar_rostro --identidad Nombre --fuente 0 --muestras 40
python principal.py --modo registrar_reid --identidad Nombre --fuente 0 --muestras 40
python principal.py --modo registrar --identidad Nombre --fuente 0 --muestras 40
python principal.py --modo capturar_rostros_video --fuente ruta\video.mp4 --intervalo 5 --muestras 50
python principal.py --modo demo --sin-ventana
```

Si se ejecuta `--modo inferir` sin fuente y esta activo el modo multicamara, intenta abrir hasta 4 camaras configuradas.

## Menu principal

Opciones principales:

- Inferir 4 camaras: muestra una cuadricula con hasta 4 camaras al mismo tiempo.
- Inferir camara individual: usa una webcam, URL IP, RTSP o fuente configurada.
- Cargar imagen: procesa una imagen y guarda el resultado en `salidas/`.
- Cargar video: reproduce video con inferencia, pantalla principal y panel separado.
- Entrenar modelos: entrena SVM rostro y SVM Re-ID.
- Ver estado: muestra conteos de datasets y modelos.
- Diagnostico rostro: genera metricas y matriz de confusion del modelo facial.
- Registrar rostro: captura rostros para `datos/rostros/`.
- Registrar Re-ID cuerpo/ropa: captura cuerpo completo para Re-ID.
- Registrar rostro + Re-ID: captura ambos tipos.
- Actualizar reportes: regenera diagnosticos.
- Capturar ROI rostros desde video: detecta rostros en un video y los guarda en `datos/rostros_v2/` para clasificacion manual.

## Entrenamiento

El entrenamiento esta en:

```text
sistema_reid/captura_tiempo_real.py
```

Funcion principal:

```python
entrenar_desde_capturas()
```

### Rostro

Usa:

```text
datos/rostros/<identidad>/
```

Proceso:

1. Lee imagenes por identidad.
2. Detecta ROI facial.
3. Si `omitir_rostros_sin_roi` es `false`, usa imagen completa cuando no detecta ROI.
4. Redimensiona el rostro segun `caracteristicas.tamano_rostro`.
5. Extrae HoG.
6. Entrena `modelos/svm_rostro.pkl`.

Importante: si se cambia `tamano_rostro`, hay que reentrenar. El entrenamiento y la inferencia deben usar la misma medida.

### Re-ID

Usa:

```text
datos/reidentificacionF/<identidad>/
```

Proceso:

1. Lee imagenes de cuerpo completo/ropa.
2. Extrae HSV espacial.
3. Usa histograma global y 3 bandas horizontales.
4. Aplica augmentation si esta activo.
5. Entrena `modelos/svm_reidentificacion.pkl`.

El descriptor Re-ID actual es:

```yaml
descriptor_reid: hsv_espacial
reid_bandas_horizontales: 3
```

## Validacion 80/20

Esta configurado en:

```yaml
entrenamiento:
  validacion: 0.20
  validacion_por_clase: true
```

Esto significa 80% entrenamiento y 20% validacion. Lo recomendado es hacerlo por clase, porque evita que una identidad quede sin datos suficientes en entrenamiento o validacion.

## Inferencia

El motor principal esta en:

```text
sistema_reid/inferencia.py
```

Funciones importantes:

- `MotorInferencia.cargar_modelos()`: carga SVM facial, SVM Re-ID y modelo Re-ID combinado si existe.
- `MotorInferencia.procesar_frame()`: procesa cada frame analizado.
- `MotorInferencia.decidir_identidad()`: decide si usar rostro o Re-ID.
- `_detectar_rostro_con_zoom()`: segundo intento de rostro en la parte superior de la persona.
- `_clasificar_reid()`: fallback cuando rostro no sirve.
- `dibujar_resultados()`: dibuja cajas, nombre, metodo y porcentaje.

Ahora las etiquetas muestran porcentaje junto al nombre:

```text
Nombre Persona 86.4% | rostro_hog_svm
Nombre Persona 64.2% | reid_hsv_svm
```

Tambien el panel lateral muestra porcentajes en `Prediccion actual` y en el ranking.

## Video fluido

Configuracion:

```yaml
video:
  procesar_cada_n_frames: 2
  modo_fluido: true
  saltar_frames_atrasados: true
  max_desfase_roi_frames: 0
```

Significado:

- `procesar_cada_n_frames`: cada cuantos frames intenta analizar con IA.
- `modo_fluido`: no congela el video esperando la IA; reutiliza el ultimo resultado disponible.
- `saltar_frames_atrasados`: si el procesamiento se atrasa, salta frames para recuperar fluidez.
- `max_desfase_roi_frames: 0`: no oculta cajas por desfase, para evitar que desaparezcan al final del video.

## Visualizacion de cajas

Configuracion:

```yaml
visualizacion:
  modo_cajas: rostro
```

Opciones:

- `persona`: solo caja grande de persona.
- `rostro`: ROI de rostro si existe; si no existe, muestra caja de persona.
- `ambas`: persona + ROI de rostro.

## Re-ID en vivo

Configuracion:

```yaml
aprendizaje_reid_en_vivo:
  activo: true
  combinar_con_reid_fijo: true
  guardar_capturas: true
  carpeta_capturas: datos/reidentificacionF
  minimo_por_identidad: 3
  reentrenar_cada: 4
  modelo_salida: svm_reidentificacion_combinado.pkl
  max_muestras_vivas_por_identidad: 100
  max_imagenes_carpeta_por_identidad: 100
```

Cuando el rostro se reconoce con suficiente confianza, el sistema guarda el cuerpo completo en `datos/reidentificacionF/<identidad>/` y agrega su descriptor HSV al entrenamiento vivo.

Diferencia entre limites:

- `max_muestras_vivas_por_identidad`: maximo de vectores HSV en memoria/buffer para reentrenar en vivo.
- `max_imagenes_carpeta_por_identidad`: maximo de imagenes reales guardadas en disco por identidad.

## Datasets y carpetas

```text
datos/rostros/                 dataset facial usado para HoG + SVM rostro
datos/rostros_v2/              rostros extraidos de videos para clasificacion manual
datos/reidentificacionF/       dataset Re-ID automatico/controlado de cuerpo completo
modelos/                       modelos SVM y YOLO
registros/                     logs CSV y metadata
reportes/                      diagnosticos y matriz de confusion
salidas/                       imagenes procesadas
pruebas/                       videos e imagenes de prueba
```

## GitHub sin subir datasets

El `.gitignore` conserva estructura y evita subir datos pesados o privados.

Se ignoran contenidos de:

```text
datos/rostros/**
datos/reidentificacionF/**
datos/rostros_v2/**
modelos/**
registros/**
reportes/**
salidas/**
pruebas/videos/**
```

Se dejan `.gitkeep` o README donde corresponde para que la estructura exista sin subir imagenes reales.

## Variables importantes del YAML

```yaml
caracteristicas:
  tamano_rostro: [96, 96]
```

Tamano al que se normaliza el ROI facial antes de HoG. Debe coincidir entre entrenamiento e inferencia.

```yaml
umbrales:
  score_rostro: 0.70
  margen_rostro: 0.05
  score_reid: 0.60
  nitidez_minima: 30.0
  tamano_minimo_rostro: 30
```

Controlan cuando aceptar rostro, cuando pasar a Re-ID y que tan util debe ser el ROI facial.

```yaml
entrenamiento:
  kernel_rostro: linear
  kernel_reid: rbf
```

Permiten usar kernels distintos para rostro y Re-ID.

## Salida esperada

En pantalla se ven:

- Caja de persona o rostro segun `modo_cajas`.
- Nombre identificado.
- Porcentaje de confianza.
- Metodo usado: `rostro_hog_svm`, `rostro_hog_svm_zoom`, `reid_hsv_svm`, etc.
- Panel con FPS, umbrales, prediccion actual, ranking y diagnostico de entrenamiento.

## Teclas

- `q`: cerrar inferencia o captura.
- `d`: actualizar diagnostico facial durante inferencia.

