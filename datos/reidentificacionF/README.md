# ReidentificacionF - Flujo Re-ID vivo

Esta carpeta documenta el concepto del flujo de re-identificacion en vivo del proyecto.

## Idea principal

La Re-ID no empieza desde el modo de captura manual, sino durante la inferencia en vivo:

1. La camara o el video entrega frames.
2. YOLO detecta la persona.
3. El sistema busca ROI de rostro dentro de la persona.
4. Si el rostro se reconoce con suficiente confianza, esa identidad se considera confirmada.
5. Con esa identidad confirmada, el sistema toma la ROI completa de la persona.
6. La ROI completa se convierte en descriptor HSV espacial de 3 bandas.
7. La muestra se almacena en el buffer Re-ID vivo.
8. La imagen de cuerpo completo tambien se guarda en `datos/reidentificacionF/<identidad>/`.
9. Cada cierto numero de muestras nuevas, se intenta reentrenar el SVM Re-ID combinado.

## Retroalimentacion progresiva

El objetivo es que Re-ID se alimente de datos confirmados por rostro:

```text
rostro confiable -> identidad confirmada -> cuerpo completo HSV 3 bandas -> buffer vivo + imagen en reidentificacionF -> reentrenamiento Re-ID
```

Esto evita guardar cuerpo/ropa con una identidad equivocada cuando el rostro no es confiable.

El modelo combinado de inferencia usa:

```text
datos/reidentificacionF/
+ modelos/buffer_reid_hsv_en_vivo.npz
-> svm_reidentificacion_combinado.pkl
```

Si `reentrenar_combinado_al_iniciar: true`, el combinado se reconstruye al abrir la inferencia para no quedarse usando un `.pkl` viejo.

## Estructura de guardado

Las capturas nuevas se organizan por el nombre reconocido por el modelo facial:

```text
datos/reidentificacionF/
  Elias Jacome/
    Elias Jacome_reid_vivo_YYYYMMDD_HHMMSS_000_00001.jpg
  John Guerrero/
    John Guerrero_reid_vivo_YYYYMMDD_HHMMSS_000_00001.jpg
```

Estas imagenes de cuerpo completo quedan disponibles para un entrenamiento posterior con el boton `Entrenar modelos`.

## Data augmentation

Al presionar `Entrenar modelos`, el sistema puede generar variaciones desde las muestras automaticas de cuerpo completo:

```text
datos/reidentificacionF/<identidad>/imagen_real.jpg
-> augmentation
-> datos/reidentificacionF/<identidad>/imagen_real_aug_01.jpg
```

Si `cantidad_por_imagen: 1`, cada imagen real seleccionada para entrenamiento genera una variante `_aug_01`.
La carpeta respeta `max_imagenes_por_identidad`, por ejemplo:

```text
max_imagenes_por_identidad: 50
-> se conservan hasta 50 imagenes recientes por persona entre reales y aumentadas
```

Las variaciones aplicadas son controladas:

- Pequenos cambios de brillo.
- Pequenos cambios de contraste.
- Recortes leves.
- Espejado horizontal si no rompe la escena.
- Ligero ruido controlado.
- Rotacion leve.

La idea es mejorar robustez sin cambiar la identidad visual de la ropa. Para evitar una validacion falsa, la augmentation se aplica despues del split 80/20 y solo sobre el conjunto de entrenamiento.

Variables principales:

```yaml
augmentation_reidF:
  activo: true
  cantidad_por_imagen: 1
  guardar_imagenes: true
  max_imagenes_por_identidad: 50
```

Las imagenes aumentadas se guardan en la misma carpeta de identidad con `_aug_` en el nombre. Si se supera el limite configurado, se conservan las imagenes mas recientes.

## Descriptor Re-ID actual

La Re-ID usa HSV espacial con 3 bandas horizontales configuradas en `configuracion.yaml`:

```yaml
caracteristicas:
  descriptor_reid: hsv_espacial
  reid_bandas_horizontales: 3
```

Eso significa que el sistema no mira solo el color global de la ropa; tambien separa la persona completa en tres zonas horizontales:

```text
banda 1: zona superior
banda 2: zona media
banda 3: zona inferior / piernas
```

Asi captura mejor la distribucion de color en el cuerpo completo.

## Nota sobre nombres largos

Los nombres largos en las cajas de inferencia se recortan visualmente para evitar que el texto invada otras detecciones o se monte sobre la imagen. El nombre completo se conserva internamente en logs y ranking.
