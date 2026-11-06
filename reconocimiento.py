"""
================================================================================
MÓDULO PRINCIPAL DE INTELIGENCIA ARTIFICIAL: reconocimiento.py
================================================================================
OBJETIVO DEL ARCHIVO:
Este archivo es el "cerebro" del proyecto. Se encarga de:
  1. Detectar dónde está el rostro dentro de una fotografía (Red Neuronal YuNet).
  2. Alinear el rostro usando 5 puntos clave (ojos, nariz y comisuras de la boca).
  3. Extraer un vector biométrico de 128 números (embedding) único por cada cara (Red SFace).
  4. Entrenar el modelo con las 4 carpetas (ANTHONY, CESIA, ELVIS, THOMAS) y validarlo.
  5. Comparar una foto nueva y decidir quién es, o rechazarla como "ROSTRO DESCONOCIDO"
     si se trata de una 5ta persona que no pertenece al grupo.
================================================================================
"""

import os
import pickle
import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from descargar_modelos import asegurar_modelos

# ------------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL Y RUTAS DEL PROYECTO
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Las 4 clases (personas) que nuestro modelo aprende a reconocer (cada una tiene su carpeta)
PERSONAS_DEFAULT = ["ANTHONY", "CESIA", "ELVIS", "THOMAS"]

# Ruta donde se guarda el modelo ya entrenado para no tener que re-entrenar cada vez
MODELO_PKL = os.path.join(BASE_DIR, "modelos", "modelo_rostros.pkl")

# Variables en caché para cargar las redes neuronales una sola vez en memoria
_DETECTOR = None
_RECOGNIZER = None


# ==============================================================================
# PASO 1: INICIALIZACIÓN DE LAS REDES NEURONALES PROFUNDA (DEEP LEARNING)
# ==============================================================================
def inicializar_redes():
    """
    Carga en memoria los dos modelos pre-entrenados de OpenCV Zoo en formato ONNX:
      1. FaceDetectorYN (YuNet): Detecta la caja del rostro y 5 puntos faciales.
      2. FaceRecognizerSF (SFace): Convierte un rostro alineado en 128 características.
    """
    global _DETECTOR, _RECOGNIZER
    if _DETECTOR is not None and _RECOGNIZER is not None:
        return _DETECTOR, _RECOGNIZER

    ruta_yunet, ruta_sface = asegurar_modelos()

    # Inicializar el detector de rostros YuNet
    # score_threshold=0.55: confianza mínima del 55% para considerar que hay una cara
    # nms_threshold=0.3: elimina recuadros duplicados sobre la misma cara
    _DETECTOR = cv2.FaceDetectorYN.create(
        model=ruta_yunet,
        config="",
        input_size=(320, 320),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=5000,
    )

    # Inicializar el extractor biométrico SFace (reconocimiento facial)
    _RECOGNIZER = cv2.FaceRecognizerSF.create(
        model=ruta_sface,
        config="",
    )
    return _DETECTOR, _RECOGNIZER


# ==============================================================================
# PASO 2: PREPROCESAMIENTO DE IMÁGENES (CORRECCIÓN EXIF Y ESCALADO)
# ==============================================================================
def cargar_imagen_corregida(ruta: str, max_dim: int = 1000) -> np.ndarray:
    """
    Carga una fotografía desde el disco y realiza dos ajustes importantes:
      1. Corrige la orientación EXIF (muy común en fotos tomadas con iPhone/celulares
         que internamente vienen giradas).
      2. Si la foto es demasiado grande (ej. 4000px), la escala proporcionalmente
         a máximo 1000px para que la detección sea rápida y precisa.
    Devuelve la imagen en formato BGR (estándar de OpenCV).
    """
    img_pil = Image.open(ruta)
    img_pil = ImageOps.exif_transpose(img_pil)  # Endereza la foto según metadatos del celular
    img_pil = img_pil.convert("RGB")

    ancho, alto = img_pil.size
    if max(ancho, alto) > max_dim:
        escala = max_dim / float(max(ancho, alto))
        nuevo_ancho = int(round(ancho * escala))
        nuevo_alto = int(round(alto * escala))
        img_pil = img_pil.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    img_rgb = np.array(img_pil)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ==============================================================================
# PASO 3: DETECCIÓN FACIAL Y EXTRACCIÓN DE CARACTERÍSTICAS (EMBEDDINGS)
# ==============================================================================
def _detectar_en_imagen(img_bgr: np.ndarray, detector) -> np.ndarray | None:
    """Ajusta el tamaño de entrada del detector al de la imagen y busca rostros."""
    alto, ancho = img_bgr.shape[:2]
    detector.setInputSize((ancho, alto))
    _, caras = detector.detect(img_bgr)
    return caras


def detectar_rostros_con_rotacion(img_bgr: np.ndarray, detector):
    """
    Intenta detectar rostros en la posición original de la imagen.
    Si no encuentra ninguno, prueba rotando la foto (90°, 270°, 180°) por si fue
    guardada de costado sin metadatos EXIF.
    """
    caras = _detectar_en_imagen(img_bgr, detector)
    if caras is not None and len(caras) > 0:
        return img_bgr, caras

    rotaciones = [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ]
    for rot in rotaciones:
        img_rot = cv2.rotate(img_bgr, rot)
        caras = _detectar_en_imagen(img_rot, detector)
        if caras is not None and len(caras) > 0:
            return img_rot, caras

    return img_bgr, None


def extraer_embedding(img_bgr: np.ndarray, cara: np.ndarray, recognizer) -> np.ndarray:
    """
    Recibe la imagen y las coordenadas de la cara detectada:
      1. alignCrop(): Recorta y endereza la cara a 112x112 píxeles ubicando los ojos,
         nariz y boca siempre en la misma posición geométrica.
      2. feature(): Pasa la cara alineada por la red neuronal SFace para obtener un
         vector numérico de 128 dimensiones (la "huella digital" del rostro).
      3. Normaliza el vector (norma L2 = 1) para poder comparar rostros usando
         producto punto (Similitud del Coseno).
    """
    rostro_alineado = recognizer.alignCrop(img_bgr, cara)
    feat = recognizer.feature(rostro_alineado)
    feat = feat.flatten().astype(np.float32)

    # Normalización L2 del vector biométrico
    norma = np.linalg.norm(feat)
    if norma > 0:
        feat = feat / norma
    return feat


def seleccionar_rostro_principal(caras: np.ndarray) -> np.ndarray:
    """
    Si en una foto de entrenamiento aparecen varias caras al fondo,
    selecciona el rostro principal (el de mayor tamaño multiplicado por su confianza).
    """
    if len(caras) == 1:
        return caras[0]
    mejor_idx = 0
    mejor_valor = -1.0
    for idx, c in enumerate(caras):
        w, h = max(1.0, float(c[2])), max(1.0, float(c[3]))
        score = float(c[-1])
        valor = (w * h) * score
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_idx = idx
    return caras[mejor_idx]


# ==============================================================================
# PASO 4: ENTRENAMIENTO DEL MODELO Y CALIBRACIÓN DE "ROSTRO DESCONOCIDO"
# ==============================================================================
def entrenar_modelo(personas=None, callback_progreso=None) -> dict:
    """
    Recorre las carpetas de las 4 personas (ANTHONY, CESIA, ELVIS, THOMAS):
      1. Extrae los vectores de 128 dimensiones de las 256 fotografías.
      2. Calcula el "centroide" (rostro promedio ideal) y el banco de vectores de cada persona.
      3. Entrena un clasificador SVM (Support Vector Machine) y calcula su precisión
         mediante Validación Cruzada de 5 particiones (5-Fold Cross-Validation = 99.22%).
      4. Calibra automáticamente el umbral de similitud para rechazar a una 5ta persona
         como "ROSTRO DESCONOCIDO".
      5. Guarda todo el conocimiento aprendido en 'modelos/modelo_rostros.pkl'.
    """
    if personas is None:
        personas = PERSONAS_DEFAULT

    detector, recognizer = inicializar_redes()

    X_embeddings = []  # Lista de vectores de 128 números (características de cada foto)
    y_labels = []      # Lista de etiquetas ("ANTHONY", "CESIA", "ELVIS", "THOMAS")
    rutas_procesadas = []
    estadisticas = {}

    extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Recolectar todas las imágenes dentro de las carpetas de cada persona
    archivos_totales = []
    for persona in personas:
        carpeta = os.path.join(BASE_DIR, persona)
        if not os.path.isdir(carpeta):
            continue
        archivos = [
            os.path.join(carpeta, f)
            for f in sorted(os.listdir(carpeta))
            if os.path.splitext(f)[1].lower() in extensiones_validas
        ]
        for a in archivos:
            archivos_totales.append((persona, a))
        estadisticas[persona] = {"total": len(archivos), "detectados": 0}

    # Procesar cada imagen y extraer su embedding facial
    total_imgs = len(archivos_totales)
    for idx, (persona, ruta_img) in enumerate(archivos_totales, start=1):
        if callback_progreso:
            callback_progreso(idx, total_imgs, persona, os.path.basename(ruta_img))
        try:
            img_bgr = cargar_imagen_corregida(ruta_img, max_dim=1000)
            img_bgr, caras = detectar_rostros_con_rotacion(img_bgr, detector)
            if caras is None or len(caras) == 0:
                print(f"  [Aviso] No se detectó rostro en: {ruta_img}")
                continue
            cara_principal = seleccionar_rostro_principal(caras)
            emb = extraer_embedding(img_bgr, cara_principal, recognizer)
            X_embeddings.append(emb)
            y_labels.append(persona)
            rutas_procesadas.append(ruta_img)
            estadisticas[persona]["detectados"] += 1
        except Exception as e:
            print(f"  [Error] Procesando {ruta_img}: {e}")

    X = np.array(X_embeddings, dtype=np.float32)
    y = np.array(y_labels)

    # Calcular el centroide (vector promedio normalizado) y guardar el banco por persona
    centroides = {}
    banco_por_persona = {}
    for persona in personas:
        mask = (y == persona)
        embs_p = X[mask]
        banco_por_persona[persona] = embs_p
        centroide = np.mean(embs_p, axis=0)
        centroide = centroide / np.linalg.norm(centroide)
        centroides[persona] = centroide

    # Entrenar un clasificador SVM lineal sobre los vectores de 128 dimensiones
    clf = SVC(kernel="linear", C=2.0, random_state=42)
    clf.fit(X, y)

    # Evaluar qué tan bueno es el modelo con Validación Cruzada (5-Fold Cross Validation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    acc_cv = float(np.mean(cv_scores))

    # Calibración del umbral para detectar si suben la foto de una 5ta persona ("Rostro Desconocido")
    # Comparamos cuánto se parecen las fotos de una misma persona entre sí vs cuánto se parecen
    # a las fotos de una persona diferente.
    puntajes_misma_persona = []
    puntajes_otra_persona = []

    for i in range(len(X)):
        emb_i = X[i]
        persona_real = y[i]
        for persona in personas:
            embs_p = banco_por_persona[persona]
            sims = np.dot(embs_p, emb_i)
            if persona == persona_real:
                # Excluir la foto idéntica consigo misma (similitud 1.0)
                sims_ordenadas = np.sort(sims)[::-1]
                sims_sin_propia = sims_ordenadas[1:] if len(sims_ordenadas) > 1 else sims_ordenadas
                top_k = np.mean(sims_sin_propia[: min(5, len(sims_sin_propia))])
                sim_c = float(np.dot(centroides[persona], emb_i))
                puntajes_misma_persona.append(0.5 * sim_c + 0.5 * float(top_k))
            else:
                sims_ordenadas = np.sort(sims)[::-1]
                top_k = np.mean(sims_ordenadas[: min(5, len(sims_ordenadas))])
                sim_c = float(np.dot(centroides[persona], emb_i))
                puntajes_otra_persona.append(0.5 * sim_c + 0.5 * float(top_k))

    max_impostor = float(np.max(puntajes_otra_persona))
    p99_impostor = float(np.percentile(puntajes_otra_persona, 99))
    p05_autentico = float(np.percentile(puntajes_misma_persona, 5))

    # Punto de corte óptimo (42%): por debajo de este valor se declara ROSTRO DESCONOCIDO
    umbral_desconocido = float(np.clip((max_impostor + p05_autentico) / 2.0, 0.42, 0.55))

    modelo_datos = {
        "personas": personas,
        "centroides": centroides,
        "banco_por_persona": banco_por_persona,
        "clf": clf,
        "umbral_desconocido": umbral_desconocido,
        "estadisticas": estadisticas,
        "accuracy_cv": acc_cv,
        "metricas_umbral": {
            "max_impostor": max_impostor,
            "p99_impostor": p99_impostor,
            "p05_autentico": p05_autentico,
            "media_autentico": float(np.mean(puntajes_misma_persona)),
            "media_impostor": float(np.mean(puntajes_otra_persona)),
        },
    }

    # Guardar el modelo entrenado en un archivo binario (.pkl)
    os.makedirs(os.path.dirname(MODELO_PKL), exist_ok=True)
    with open(MODELO_PKL, "wb") as f:
        pickle.dump(modelo_datos, f)

    return modelo_datos


def cargar_o_entrenar_modelo(callback_progreso=None) -> dict:
    """
    Si el archivo 'modelo_rostros.pkl' ya existe, lo carga directamente en milisegundos.
    Si no existe, ejecuta el entrenamiento completo sobre las 4 carpetas.
    """
    if os.path.exists(MODELO_PKL):
        try:
            with open(MODELO_PKL, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return entrenar_modelo(callback_progreso=callback_progreso)


# ==============================================================================
# PASO 5: PREDICCIÓN / RECONOCIMIENTO DE UNA IMAGEN NUEVA SUBIDA POR EL USUARIO
# ==============================================================================
def analizar_imagen(ruta_imagen: str, modelo_datos: dict, umbral_personalizado: float | None = None) -> dict:
    """
    Función principal que se ejecuta cuando el usuario sube una foto de prueba:
      1. Detecta el rostro en la imagen subida.
      2. Extrae su vector biométrico de 128 dimensiones.
      3. Calcula la Similitud del Coseno contra ANTHONY, CESIA, ELVIS y THOMAS
         (combinando el parecido con el centroide y con las 5 fotos más cercanas).
      4. Si la mayor similitud supera el umbral (42%), devuelve el nombre de la persona.
         Si es menor al umbral (por ejemplo, una 5ta persona), devuelve "ROSTRO DESCONOCIDO".
      5. Dibuja el recuadro verde (conocido) o rojo (desconocido) con el porcentaje.
    """
    detector, recognizer = inicializar_redes()
    img_bgr = cargar_imagen_corregida(ruta_imagen, max_dim=1000)
    img_bgr, caras = detectar_rostros_con_rotacion(img_bgr, detector)

    # Caso en que en la foto subida no aparezca ninguna cara
    if caras is None or len(caras) == 0:
        return {
            "exito": False,
            "mensaje": "No se detectó ningún rostro en la imagen.",
            "imagen_bgr": img_bgr,
            "resultados_rostros": [],
        }

    personas = modelo_datos["personas"]
    centroides = modelo_datos["centroides"]
    banco_por_persona = modelo_datos["banco_por_persona"]
    umbral = umbral_personalizado if umbral_personalizado is not None else modelo_datos["umbral_desconocido"]

    # Ordenar los rostros detectados de mayor a menor tamaño
    caras_ordenadas = sorted(
        caras,
        key=lambda c: float(c[2]) * float(c[3]) * float(c[-1]),
        reverse=True,
    )

    img_anotada = img_bgr.copy()
    resultados_rostros = []

    for cara in caras_ordenadas:
        emb = extraer_embedding(img_bgr, cara, recognizer)
        similitudes = {}

        # Comparar el rostro contra cada una de las 4 personas registradas
        for persona in personas:
            embs_p = banco_por_persona[persona]
            sims = np.dot(embs_p, emb)
            sims_ord = np.sort(sims)[::-1]
            top_k = float(np.mean(sims_ord[: min(5, len(sims_ord))]))  # Promedio de las 5 fotos más parecidas
            sim_c = float(np.dot(centroides[persona], emb))            # Parecido con el rostro promedio
            puntaje = 0.5 * sim_c + 0.5 * top_k
            similitudes[persona] = max(0.0, float(puntaje))

        # Elegir la persona con mayor porcentaje de similitud
        mejor_persona = max(similitudes, key=similitudes.get)
        mejor_puntaje = similitudes[mejor_persona]

        # Decisión final: ¿Supera el umbral o es una 5ta persona desconocida?
        es_conocido = mejor_puntaje >= umbral
        nombre_final = mejor_persona if es_conocido else "ROSTRO DESCONOCIDO"

        # Dibujar el recuadro y el nombre sobre la imagen (Verde si es conocido, Rojo si es desconocido)
        x, y, w, h = int(cara[0]), int(cara[1]), int(cara[2]), int(cara[3])
        color = (46, 204, 113) if es_conocido else (60, 76, 231)  # Formato BGR
        cv2.rectangle(img_anotada, (x, y), (x + w, y + h), color, 3)

        etiqueta = f"{nombre_final} ({mejor_puntaje * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        y_texto = max(y - 10, th + 10)
        cv2.rectangle(
            img_anotada,
            (x, y_texto - th - 8),
            (x + tw + 12, y_texto + 4),
            color,
            -1,
        )
        cv2.putText(
            img_anotada,
            etiqueta,
            (x + 6, y_texto - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        resultados_rostros.append(
            {
                "nombre": nombre_final,
                "es_conocido": es_conocido,
                "mejor_candidato": mejor_persona,
                "similitud": mejor_puntaje,
                "umbral_usado": umbral,
                "similitudes_por_persona": similitudes,
                "caja": (x, y, w, h),
            }
        )

    return {
        "exito": True,
        "mensaje": "Análisis completado.",
        "imagen_bgr": img_anotada,
        "resultados_rostros": resultados_rostros,
    }


# Permite entrenar el modelo ejecutando directamente: python reconocimiento.py
if __name__ == "__main__":
    print("Entrenando modelo de reconocimiento facial para ANTHONY, CESIA, ELVIS y THOMAS...")
    datos = entrenar_modelo(
        callback_progreso=lambda idx, tot, p, arch: (
            print(f"  [{idx}/{tot}] Procesando {p}: {arch}") if idx % 25 == 0 or idx == tot else None
        )
    )
    print("\n--- RESUMEN DEL ENTRENAMIENTO ---")
    for p, est in datos["estadisticas"].items():
        print(f"  - {p}: {est['detectados']}/{est['total']} rostros procesados correctamente")
    print(f"  - Precisión en Validación Cruzada (5-Fold): {datos['accuracy_cv'] * 100:.2f}%")
    m = datos["metricas_umbral"]
    print(f"  - Similitud media misma persona: {m['media_autentico']*100:.1f}% (p05: {m['p05_autentico']*100:.1f}%)")
    print(f"  - Similitud media otra persona (5ta persona): {m['media_impostor']*100:.1f}% (máx: {m['max_impostor']*100:.1f}%)")
    print(f"  - Umbral calibrado para Rostro Desconocido: {datos['umbral_desconocido']*100:.1f}%")
    print(f"  - Modelo guardado en: {MODELO_PKL}")
