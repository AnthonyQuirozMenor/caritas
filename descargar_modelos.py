"""
================================================================================
MÓDULO AUXILIAR DE DESCARGA DE MODELOS ONNX: descargar_modelos.py
================================================================================
Verifica que los dos modelos pre-entrenados de Deep Learning de OpenCV Zoo
estén descargados en la carpeta 'modelos/':
  1. face_detection_yunet_2023mar.onnx  -> Detector de rostros y 5 puntos clave.
  2. face_recognition_sface_2021dec.onnx -> Extractor biométrico de 128 dimensiones.
================================================================================
"""

import os
import urllib.request

MODELOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")

URLS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def asegurar_modelos() -> tuple[str, str]:
    """Garantiza que ambos archivos .onnx existan en disco y devuelve sus rutas."""
    os.makedirs(MODELOS_DIR, exist_ok=True)
    rutas = []
    for nombre, url in URLS.items():
        destino = os.path.join(MODELOS_DIR, nombre)
        if not os.path.exists(destino) or os.path.getsize(destino) < 10_000:
            print(f"Descargando {nombre}...")
            urllib.request.urlretrieve(url, destino)
            print(f"Descargado: {destino} ({os.path.getsize(destino) / (1024*1024):.2f} MB)")
        rutas.append(destino)
    return rutas[0], rutas[1]


if __name__ == "__main__":
    asegurar_modelos()
