from __future__ import annotations

import ast
import html
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
SALIDA = BASE / "info" / "Documentacion_Proyecto_ReID_HOG_HSV.docx"


def xml_escape(text: Any) -> str:
    return html.escape(str(text), quote=False)


def limpiar_docstring(texto: str | None) -> str:
    if not texto:
        return ""
    return " ".join(texto.strip().split())


def leer_configuracion() -> dict[str, Any]:
    ruta = BASE / "configuracion.yaml"
    return cargar_yaml_simple(ruta.read_text(encoding="utf-8"))


def convertir_valor_yaml(valor: str) -> Any:
    valor = valor.strip()
    if valor == "":
        return {}
    if valor.lower() == "true":
        return True
    if valor.lower() == "false":
        return False
    if valor.lower() in {"null", "none"}:
        return None
    if valor.startswith("[") and valor.endswith("]"):
        try:
            return ast.literal_eval(valor)
        except Exception:
            return valor
    try:
        if "." in valor:
            return float(valor)
        return int(valor)
    except ValueError:
        return valor.strip("\"'")


def cargar_yaml_simple(texto: str) -> dict[str, Any]:
    """Parser suficiente para la estructura de configuracion.yaml del proyecto."""
    raiz: dict[str, Any] = {}
    stack: list[tuple[int, Any, str | None]] = [(-1, raiz, None)]

    for linea in texto.splitlines():
        if not linea.strip() or linea.lstrip().startswith("#"):
            continue
        sin_comentario = linea.split("#", 1)[0].rstrip()
        if not sin_comentario.strip():
            continue
        indent = len(sin_comentario) - len(sin_comentario.lstrip(" "))
        contenido = sin_comentario.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        contenedor = stack[-1][1]

        if contenido.startswith("- "):
            item_texto = contenido[2:].strip()
            padre = stack[-1][1]
            clave_padre = stack[-1][2]
            if isinstance(padre, dict) and clave_padre and not isinstance(padre.get(clave_padre), list):
                padre[clave_padre] = []
                contenedor = padre[clave_padre]
            elif isinstance(padre, list):
                contenedor = padre
            if not isinstance(contenedor, list):
                continue
            if ":" in item_texto:
                clave, valor = item_texto.split(":", 1)
                nuevo: dict[str, Any] = {clave.strip(): convertir_valor_yaml(valor)}
                contenedor.append(nuevo)
                stack.append((indent, nuevo, clave.strip()))
            else:
                contenedor.append(convertir_valor_yaml(item_texto))
            continue

        if ":" not in contenido or not isinstance(contenedor, dict):
            continue
        clave, valor = contenido.split(":", 1)
        clave = clave.strip()
        valor_convertido = convertir_valor_yaml(valor)
        contenedor[clave] = valor_convertido
        stack.append((indent, contenedor if valor.strip() else valor_convertido, clave))
    return raiz


def listar_modulos() -> list[dict[str, Any]]:
    rutas = [BASE / "principal.py", *sorted((BASE / "sistema_reid").glob("*.py"))]
    modulos: list[dict[str, Any]] = []
    for ruta in rutas:
        if ruta.name == "__init__.py":
            continue
        try:
            arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        except Exception:
            continue
        clases = []
        funciones = []
        for nodo in arbol.body:
            if isinstance(nodo, ast.ClassDef):
                metodos = [
                    {
                        "nombre": item.name,
                        "doc": limpiar_docstring(ast.get_docstring(item)),
                    }
                    for item in nodo.body
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("__")
                ]
                clases.append(
                    {
                        "nombre": nodo.name,
                        "doc": limpiar_docstring(ast.get_docstring(nodo)),
                        "metodos": metodos,
                    }
                )
            elif isinstance(nodo, ast.FunctionDef):
                funciones.append(
                    {
                        "nombre": nodo.name,
                        "doc": limpiar_docstring(ast.get_docstring(nodo)),
                    }
                )
        modulos.append(
            {
                "ruta": str(ruta.relative_to(BASE)).replace("\\", "/"),
                "doc": limpiar_docstring(ast.get_docstring(arbol)),
                "clases": clases,
                "funciones": funciones,
            }
        )
    return modulos


def tag_p(texto: str = "", estilo: str | None = None, bold: bool = False, code: bool = False, num_id: int | None = None) -> str:
    ppr = ""
    if estilo:
        ppr += f'<w:pStyle w:val="{estilo}"/>'
    if num_id is not None:
        ppr += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
    ppr = f"<w:pPr>{ppr}</w:pPr>" if ppr else ""
    rpr = ""
    if bold:
        rpr += "<w:b/>"
    if code:
        rpr += '<w:rStyle w:val="CodeChar"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t xml:space=\"preserve\">{xml_escape(texto)}</w:t></w:r></w:p>"


def tag_page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def tag_table(rows: list[list[str]], widths: list[int] | None = None) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    if widths is None:
        widths = [int(9360 / cols)] * cols
    while len(widths) < cols:
        widths.append(int(9360 / cols))
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths[:cols])
    xml = [
        "<w:tbl>",
        (
            "<w:tblPr>"
            '<w:tblW w:w="9360" w:type="dxa"/>'
            '<w:tblInd w:w="120" w:type="dxa"/>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="4" w:space="0" w:color="B8C2CC"/>'
            '<w:left w:val="single" w:sz="4" w:space="0" w:color="B8C2CC"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="B8C2CC"/>'
            '<w:right w:val="single" w:sz="4" w:space="0" w:color="B8C2CC"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D7DEE8"/>'
            '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D7DEE8"/>'
            "</w:tblBorders>"
            '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
            '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar>'
            "</w:tblPr>"
        ),
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for i, row in enumerate(rows):
        xml.append("<w:tr>")
        for j in range(cols):
            texto = row[j] if j < len(row) else ""
            fill = '<w:shd w:fill="E8EEF5"/>' if i == 0 else ""
            bold = i == 0
            xml.append(
                "<w:tc>"
                f'<w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{fill}</w:tcPr>'
                f'{tag_p(texto, bold=bold)}'
                "</w:tc>"
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def tag_callout(titulo: str, cuerpo: str) -> str:
    return tag_table([[titulo], [cuerpo]], [9360])


def flatten_config(prefix: str, value: Any, rows: list[list[str]]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_config(f"{prefix}.{key}" if prefix else str(key), nested, rows)
    elif isinstance(value, list):
        rows.append([prefix, json.dumps(value, ensure_ascii=False), "Lista/estructura definida en YAML."])
    else:
        rows.append([prefix, str(value), descripcion_config(prefix)])


def descripcion_config(clave: str) -> str:
    descripciones = {
        "ejecucion.dispositivo": "Selecciona CPU/GPU si el entorno lo soporta.",
        "ejecucion.mostrar_ventana": "Muestra ventanas OpenCV durante inferencia o registro.",
        "rendimiento.reservar_nucleos": "Nucleos libres para no colapsar Windows.",
        "multicamara.activo_sin_fuente_en_inferir": "Abre 4 camaras si se ejecuta inferir sin fuente.",
        "video.procesar_cada_n_frames": "Reduce carga procesando inferencia cada N frames.",
        "visualizacion.modo_cajas": "persona, ambas o rostro para decidir que ROI se dibuja.",
        "yolo.confianza": "Score minimo de deteccion de persona.",
        "umbrales.score_rostro": "Probabilidad minima para aceptar SVM facial.",
        "umbrales.margen_rostro": "Diferencia minima entre primer y segundo candidato facial.",
        "umbrales.score_reid": "Probabilidad minima para aceptar Re-ID por HSV.",
        "umbrales.nitidez_minima": "Varianza minima del Laplaciano para usar el rostro.",
        "umbrales.tamano_minimo_rostro": "Tamano minimo del ROI facial.",
        "rostro_en_persona.usar_zoom_si_no_detecta": "Segundo intento de rostro en zona superior de persona.",
        "caracteristicas.tamano_rostro": "Tamano fijo del ROI facial para HOG.",
        "caracteristicas.tamano_torso": "Tamano fijo del torso para HSV.",
        "caracteristicas.bins_hsv": "Resolucion del histograma HSV.",
        "aprendizaje_reid_en_vivo.combinar_con_reid_fijo": "Une dataset fijo con buffer vivo antes de entrenar.",
        "aprendizaje_reid_en_vivo.reentrenar_cada": "Muestras nuevas necesarias para intentar actualizar SVM Re-ID.",
        "aprendizaje_reid_en_vivo.minimo_por_identidad": "Minimo por clase para entrenar SVM Re-ID vivo.",
        "aprendizaje_reid_en_vivo.score_rostro_min_aprendizaje": "Score facial requerido para guardar torso en vivo.",
        "aprendizaje_reid_en_vivo.margen_rostro_min_aprendizaje": "Margen facial requerido para guardar torso en vivo.",
        "entrenamiento.validacion": "Proporcion reservada para validacion.",
        "entrenamiento.validacion_por_clase": "Split estratificado por identidad.",
        "entrenamiento.omitir_rostros_sin_roi": "Controla si se omiten imagenes sin ROI facial en entrenamiento.",
    }
    return descripciones.get(clave, "")


def seccion_configuracion(config: dict[str, Any]) -> str:
    rows = [["Clave YAML", "Valor actual", "Para que sirve"]]
    flatten_config("", config, rows)
    return tag_table(rows, [3000, 2200, 4160])


def seccion_modulos(modulos: list[dict[str, Any]]) -> str:
    rows = [["Modulo", "Responsabilidad", "Clases principales", "Funciones principales"]]
    for modulo in modulos:
        clases = ", ".join(c["nombre"] for c in modulo["clases"]) or "-"
        funciones = ", ".join(f["nombre"] for f in modulo["funciones"][:8])
        if len(modulo["funciones"]) > 8:
            funciones += ", ..."
        rows.append([modulo["ruta"], modulo["doc"] or "-", clases, funciones or "-"])
    return tag_table(rows, [1800, 3500, 1800, 2260])


def seccion_funciones_detalle(modulos: list[dict[str, Any]]) -> str:
    partes: list[str] = []
    for modulo in modulos:
        partes.append(tag_p(modulo["ruta"], "Heading2"))
        if modulo["doc"]:
            partes.append(tag_p(modulo["doc"]))
        rows = [["Elemento", "Tipo", "Descripcion"]]
        for clase in modulo["clases"]:
            rows.append([clase["nombre"], "Clase", clase["doc"] or "-"])
            for metodo in clase["metodos"]:
                rows.append([f"  {metodo['nombre']}", "Metodo", metodo["doc"] or "-"])
        for funcion in modulo["funciones"]:
            rows.append([funcion["nombre"], "Funcion", funcion["doc"] or "-"])
        partes.append(tag_table(rows, [2500, 1200, 5660]))
    return "".join(partes)


def build_document() -> str:
    config = leer_configuracion()
    modulos = listar_modulos()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    body: list[str] = []
    body.append(tag_p("Documentacion tecnica para defensa", "Title"))
    body.append(tag_p("Sistema de Identificacion Facial y Re-Identificacion HOG/HSV/SVM", "Subtitle"))
    body.append(tag_p(f"Proyecto: {config.get('proyecto', {}).get('nombre', 'Proyecto Re-ID')}"))
    body.append(tag_p(f"Generado: {fecha}"))
    body.append(tag_callout("Resumen para defender", "El sistema detecta personas con YOLO, busca el rostro dentro del ROI de persona, identifica con HOG + SVM si el rostro es confiable y usa Re-ID HSV + SVM como respaldo cuando el rostro no aparece, esta borroso o no supera los umbrales. El Re-ID puede entrenarse desde carpetas fijas y actualizarse en vivo combinando esos datos con muestras nuevas etiquetadas por reconocimiento facial confiable."))
    body.append(tag_page_break())

    body.append(tag_p("1. Objetivo del sistema", "Heading1"))
    body.append(tag_p("El objetivo es reconocer personas en una PC usando camaras, videos o imagenes. La identidad se decide primero por rostro cuando existe una senal facial confiable. Si el rostro falla, el sistema recurre a una re-identificacion por apariencia del torso/ropa."))
    body.append(tag_p("El diseno respeta la regla de documentacion: deteccion de persona primero, ROI de rostro dentro de persona y Re-ID como respaldo.", num_id=1))
    body.append(tag_p("El proyecto no usa perfiles manuales como clasificador final; los dos flujos principales terminan en SVM.", num_id=1))
    body.append(tag_p("El entrenamiento y la validacion son reproducibles mediante configuracion YAML.", num_id=1))

    body.append(tag_p("2. Arquitectura general", "Heading1"))
    body.append(tag_table([
        ["Capa", "Archivo/modulo", "Responsabilidad"],
        ["Entrada", "principal.py", "Menu, consola, imagen, video, webcam, multicamara, logs y orquestacion."],
        ["Deteccion", "sistema_reid/deteccion.py", "YOLOv8n para personas y Haar Cascade para rostros dentro del ROI."],
        ["Caracteristicas", "sistema_reid/caracteristicas.py", "HOG para rostro, HSV para torso/ropa, recorte de torso y filtros de nitidez."],
        ["Modelos", "sistema_reid/modelos_svm.py", "SVM, StandardScaler, validacion estratificada, guardado/carga de artefactos."],
        ["Inferencia", "sistema_reid/inferencia.py", "Prioridad rostro, fallback Re-ID, zoom opcional, visualizacion de cajas."],
        ["Entrenamiento", "sistema_reid/captura_tiempo_real.py", "Registro, dataset, split 80/20, entrenamiento y resumen de reportes."],
        ["Re-ID vivo", "sistema_reid/reid_en_vivo.py", "Buffer HSV vivo y modelo combinado con dataset fijo."],
        ["Reportes", "sistema_reid/evaluacion.py", "Accuracy, F1, matriz de confusion y diagnostico facial."],
        ["Panel", "sistema_reid/panel_control.py", "Sliders de umbral, FPS, diagnostico y scroll."],
    ], [1500, 2600, 5260]))

    body.append(tag_p("3. Flujo de inferencia", "Heading1"))
    for paso in [
        "Se abre una fuente: webcam, video, imagen, URL HTTP/RTSP o multicamara.",
        "YOLO detecta personas y devuelve una caja de persona con su ROI.",
        "Dentro de esa persona se busca el rostro principal. El sistema ya no busca rostros en todo el frame para evitar falsos positivos en paredes u objetos.",
        "Si no hay rostro, opcionalmente se recorta la parte superior de la persona, se agranda y se vuelve a buscar rostro.",
        "Si el rostro existe y es util, se extrae HOG y se clasifica con SVM facial.",
        "Si el rostro supera score y margen, se acepta la identidad por rostro.",
        "Si el rostro no existe, es borroso, pequeno o el SVM facial no supera umbrales, se usa Re-ID con HSV del torso.",
        "Si el rostro fue reconocido con mucha confianza, el torso alimenta el Re-ID combinado en vivo.",
    ]:
        body.append(tag_p(paso, num_id=2))

    body.append(tag_p("4. Modelos y archivos generados", "Heading1"))
    body.append(tag_table([
        ["Archivo", "Tipo", "Como se crea", "Uso"],
        ["modelos/svm_rostro.pkl", "SVM HOG rostro", "Boton/menu Entrenar modelos", "Identificacion cuando se ve la cara."],
        ["modelos/svm_reidentificacion.pkl", "SVM HSV Re-ID fijo", "Boton/menu Entrenar modelos con datos/reidentificacion", "Reconocer torso/ropa cuando no hay rostro."],
        ["modelos/buffer_reid_hsv_en_vivo.npz", "Vectores HSV vivos", "Inferencia con rostro reconocido confiablemente", "Memoria de torsos nuevos etiquetados por rostro."],
        ["modelos/svm_reidentificacion_combinado.pkl", "SVM HSV combinado", "Inferencia en vivo al acumular muestras", "Combina dataset fijo + buffer vivo."],
        ["registros/metadata_entrenamiento.csv", "CSV", "Entrenamiento", "Muestras usadas para entrenar."],
        ["registros/metadata_validacion.csv", "CSV", "Entrenamiento", "Muestras reservadas para validar."],
        ["reportes/resumen_entrenamiento.json", "JSON", "Entrenamiento", "Conteos y metricas usadas por el panel."],
        ["reportes/diagnostico_rostro.*", "TXT/JSON/CSV/JPG", "Diagnostico", "Accuracy, F1 y matriz de confusion."],
    ], [2600, 1800, 2500, 2460]))

    body.append(tag_p("5. Entrenamiento", "Heading1"))
    body.append(tag_p("El boton Entrenar modelos entrena los modelos fijos de rostro y Re-ID. Primero se construyen vectores y etiquetas; despues se aplica split de validacion si esta activo."))
    body.append(tag_table([
        ["Dataset", "Carpeta", "Descriptor", "Modelo"],
        ["Rostro", "datos/rostros/<identidad>/", "HOG sobre ROI facial", "svm_rostro.pkl"],
        ["Re-ID", "datos/reidentificacion/<identidad>/", "Histograma HSV torso/ropa", "svm_reidentificacion.pkl"],
    ], [1800, 2900, 2700, 1960]))
    body.append(tag_p("Split 80/20", "Heading2"))
    body.append(tag_p("Con validacion: 0.20 y validacion_por_clase: true, cada identidad conserva aproximadamente 80% para entrenamiento y 20% para validacion. Esto evita que el split global deje clases pequenas sin entrenamiento. Si validacion es 0.0, todo se usa para entrenar."))
    body.append(tag_p("Condiciones minimas", "Heading2"))
    body.append(tag_p("Un SVM de identidad necesita al menos dos identidades y muestras suficientes por identidad. Si no se cumple, el sistema avisa y no inventa modelos.", num_id=1))
    body.append(tag_p("Para Re-ID conviene tener fotos de torso/cuerpo con frente, lados, espalda, distancia media y luz similar a la prueba.", num_id=1))
    body.append(tag_p("Para rostro conviene usar fotos claras, de frente o semi-frente, con ROI facial detectable.", num_id=1))

    body.append(tag_p("6. Re-ID fijo, en vivo y combinado", "Heading1"))
    body.append(tag_p("El Re-ID fijo se entrena con datos/reidentificacion. El Re-ID en vivo no aprende de desconocidos: solo guarda torso cuando el rostro ya fue reconocido con alta confianza. El modelo combinado une la base fija con las muestras vivas y reentrena un SVM nuevo."))
    body.append(tag_table([
        ["Modo", "Datos usados", "Ventaja", "Riesgo/control"],
        ["Fijo", "Carpetas datos/reidentificacion", "Estable y defendible", "No se adapta a cambios de ropa sin reentrenar."],
        ["Vivo", "Buffer HSV etiquetado por rostro", "Se adapta a camara y ropa actual", "Puede contaminarse si el rostro se equivoca."],
        ["Combinado", "Dataset fijo + buffer vivo", "Mantiene base estable y mejora con datos reales", "Usa umbrales altos de aprendizaje para protegerse."],
    ], [1500, 2600, 2600, 2660]))

    body.append(tag_p("7. Umbrales principales", "Heading1"))
    um = config.get("umbrales", {})
    body.append(tag_table([
        ["Variable", "Valor actual", "Significado", "Como defenderla"],
        ["score_rostro", str(um.get("score_rostro", "")), "Probabilidad minima del SVM facial.", "Evita aceptar una identidad facial con poca confianza."],
        ["margen_rostro", str(um.get("margen_rostro", "")), "Diferencia entre primer y segundo candidato.", "Evita aceptar si dos personas estan muy cerca en probabilidad."],
        ["score_reid", str(um.get("score_reid", "")), "Probabilidad minima del SVM Re-ID.", "Evita reconocer por ropa si el modelo no esta seguro."],
        ["nitidez_minima", str(um.get("nitidez_minima", "")), "Filtro de enfoque del rostro.", "Evita usar rostros borrosos."],
        ["tamano_minimo_rostro", str(um.get("tamano_minimo_rostro", "")), "Tamano minimo del ROI facial.", "Evita clasificar caras demasiado pequenas."],
    ], [1900, 1300, 3100, 3060]))

    body.append(tag_p("8. Configuracion YAML completa", "Heading1"))
    body.append(tag_p("Todas las decisiones ajustables se concentran en configuracion.yaml. Las rutas relativas se resuelven desde la carpeta del proyecto."))
    body.append(seccion_configuracion(config))

    body.append(tag_p("9. Comandos y menu", "Heading1"))
    body.append(tag_table([
        ["Comando/modo", "Uso"],
        ["python principal.py", "Abre el menu principal Tkinter si esta disponible; si no, consola."],
        ["python principal.py --modo menu", "Menu principal."],
        ["python principal.py --modo entrenar", "Entrena SVM rostro y SVM Re-ID fijo."],
        ["python principal.py --modo inferir --fuente 0", "Inferencia con webcam local."],
        ["python principal.py --modo inferir --fuente ruta/video.mp4", "Inferencia sobre video."],
        ["python principal.py --modo inferir --fuente ruta/imagen.jpg", "Inferencia sobre imagen."],
        ["python principal.py --modo inferir", "Si esta activo, abre multicamara sin seleccionar fuente."],
        ["python principal.py --modo registrar_rostro --identidad NOMBRE", "Captura rostros para una identidad."],
        ["python principal.py --modo registrar_reid --identidad NOMBRE", "Captura torso/ropa para Re-ID."],
        ["python principal.py --modo registrar --identidad NOMBRE", "Captura rostro y Re-ID en la misma sesion."],
        ["python principal.py --modo diagnostico", "Genera reportes y matriz de confusion de rostro."],
        ["python principal.py --modo revisar", "Muestra conteo de datasets, modelos y rutas."],
    ], [3300, 6060]))

    body.append(tag_p("10. Menu grafico", "Heading1"))
    body.append(tag_p("El menu Tkinter permite acceder de forma basica a entrenar, inferir, revisar datasets, diagnosticar, registrar rostro, registrar Re-ID, registrar ambos y actualizar reportes. Es intencionalmente simple para defensa y uso practico."))

    body.append(tag_p("11. Visualizacion de cajas", "Heading1"))
    body.append(tag_p("La logica siempre usa persona primero. La variable visualizacion.modo_cajas solo cambia lo que se dibuja."))
    body.append(tag_table([
        ["Valor", "Dibujo"],
        ["persona / unificada", "Muestra la caja grande de persona."],
        ["ambas", "Muestra persona y ROI facial independiente si existe."],
        ["rostro / independiente", "Muestra solo ROI facial si existe; si no existe, usa persona."],
    ], [2500, 6860]))

    body.append(tag_p("12. Reportes y diagnostico", "Heading1"))
    body.append(tag_p("El diagnostico facial evalua el modelo con el mismo split de validacion configurado. Se generan texto, JSON, matriz CSV e imagen de matriz de confusion. El panel lateral lee resumen_entrenamiento.json y diagnostico_rostro.json para mostrar el ultimo entrenamiento."))
    body.append(tag_p("Metricas", "Heading2"))
    body.append(tag_table([
        ["Metrica", "Significado"],
        ["Accuracy", "Proporcion total de aciertos."],
        ["F1 macro", "Promedio de F1 por clase; ayuda cuando hay clases desbalanceadas."],
        ["Matriz de confusion", "Cruza identidad real contra identidad predicha para ver errores."],
        ["Ranking", "Probabilidad/score de candidatos por prediccion."],
    ], [2300, 7060]))

    body.append(tag_p("13. Modulos del codigo", "Heading1"))
    body.append(seccion_modulos(modulos))

    body.append(tag_p("14. Clases y funciones", "Heading1"))
    body.append(seccion_funciones_detalle(modulos))

    body.append(tag_p("15. Preguntas frecuentes para defensa", "Heading1"))
    preguntas = [
        ["Por que HOG para rostro?", "Porque resume bordes y gradientes del rostro en un vector fijo, adecuado para un SVM clasico."],
        ["Por que HSV para Re-ID?", "Porque el Re-ID de este proyecto se basa en apariencia de ropa/torso; HSV separa tono, saturacion y brillo y funciona bien para color."],
        ["Por que SVM?", "Porque permite clasificacion supervisada con vectores de caracteristicas y funciona con datasets pequenos/medianos."],
        ["Por que rostro primero?", "Porque la cara es mas discriminativa que la ropa; Re-ID se usa como respaldo cuando no hay rostro confiable."],
        ["Que pasa con desconocidos?", "No se entrenan solos. Si no estan en rostro ni Re-ID, deben salir como desconocidos para no contaminar el modelo."],
        ["Por que hay validacion 80/20?", "Para medir rendimiento en datos no usados directamente en entrenamiento y poder defender accuracy/F1."],
        ["Por que el Re-ID vivo exige score alto?", "Para evitar guardar ropa con identidad equivocada si el rostro predijo mal."],
        ["Por que se dejo de detectar rostro directo en todo el frame?", "Para cumplir persona primero y evitar falsos positivos en fondos, cuadros o sombras."],
    ]
    body.append(tag_table([["Pregunta", "Respuesta defendible"], *preguntas], [3100, 6260]))

    body.append(tag_p("16. Limitaciones y mejoras futuras", "Heading1"))
    for item in [
        "Haar Cascade puede fallar con rostros inclinados, poca luz o camaras lejanas; una mejora seria YuNet, MediaPipe o un detector DNN.",
        "HSV depende de ropa/color; si dos personas usan ropa muy parecida, el score Re-ID puede confundirse.",
        "La re-identificacion por ropa cambia si la persona cambia de ropa.",
        "YOLO consume CPU; por eso existen tamano_yolo, procesar_cada_n_frames e inferencia_async.",
        "Un tracker temporal mas fuerte podria estabilizar identidades entre frames.",
    ]:
        body.append(tag_p(item, num_id=1))

    body.append(tag_p("17. Checklist de defensa", "Heading1"))
    for item in [
        "Mostrar configuracion.yaml y explicar umbrales.",
        "Mostrar carpetas datos/rostros y datos/reidentificacion organizadas por identidad.",
        "Ejecutar entrenar y explicar split train/validacion.",
        "Mostrar panel con FPS, prediccion, ranking y diagnostico.",
        "Probar rostro visible y luego rostro no visible para activar Re-ID.",
        "Explicar que Re-ID combinado usa dataset fijo + buffer vivo.",
        "Mostrar reportes generados en reportes/.",
    ]:
        body.append(tag_p(item, num_id=1))

    sect = (
        '<w:sectPr>'
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return "".join(body) + sect


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:qFormat/><w:pPr><w:spacing w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/><w:color w:val="1F4D78"/><w:sz w:val="42"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:qFormat/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="666666"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="200"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="280" w:after="140"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="character" w:styleId="CodeChar"><w:name w:val="Code Char"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:color w:val="333333"/></w:rPr></w:style>
</w:styles>"""


def numbering_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="540" w:hanging="270"/></w:pPr></w:lvl></w:abstractNum>
  <w:abstractNum w:abstractNumId="2"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="540" w:hanging="270"/></w:pPr></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>
  <w:num w:numId="2"><w:abstractNumId w:val="2"/></w:num>
</w:numbering>"""


def document_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}</w:body>
</w:document>"""


def write_docx() -> None:
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    body = build_document()
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Documentacion Proyecto ReID HOG HSV SVM</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.utcnow().isoformat()}Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.utcnow().isoformat()}Z</dcterms:modified>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/></w:settings>"""

    with zipfile.ZipFile(SALIDA, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/_rels/document.xml.rels", doc_rels)
        docx.writestr("word/document.xml", document_xml(body))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/numbering.xml", numbering_xml())
        docx.writestr("word/settings.xml", settings)
        docx.writestr("docProps/core.xml", core)
        docx.writestr("docProps/app.xml", app)
    print(SALIDA)


if __name__ == "__main__":
    write_docx()
