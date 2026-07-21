# Resumen de implementación según la documentación

El proyecto queda alineado al flujo oficial:

- La PC ejecuta detección, extracción de características, entrenamiento e inferencia.
- Los celulares solo actúan como cámaras USB/IP/RTSP/HTTP o fuentes de video.
- YOLOv8n se usa solo para detectar personas y recortar ROI.
- La identificación facial se hace cuando el rostro es visible: **HoG + SVM facial**.
- La re-identificación se activa si el rostro no se ve, está borroso o el score facial es bajo: **HSV + SVM Re-ID**.
- Durante la inferencia en vivo, cuando el rostro sí es reconocido, se guarda HSV del torso con esa identidad para entrenar/reentrenar el SVM Re-ID en vivo.
- Si ningún score supera los umbrales, se muestra `desconocido` y se registra en logs.
