"""
================================================================================
EXPORTADOR DEL MODELO ENTRENADO A LA WEB: exportar_web.py
================================================================================
Toma el modelo entrenado en Python ('modelos/modelo_rostros.pkl') y exporta
los centroides y vectores de 128 dimensiones de ANTHONY, CESIA, ELVIS y THOMAS
a un archivo ligero JSON ('modelos/embeddings_web.json') para que la versión
web de GitHub Pages ('index.html') pueda reconocer rostros desde el celular.
================================================================================
"""

import json
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PKL = os.path.join(BASE_DIR, "modelos", "modelo_rostros.pkl")
JSON_OUT = os.path.join(BASE_DIR, "modelos", "embeddings_web.json")


def exportar():
    """Convierte los vectores biométricos de NumPy (.pkl) a formato JSON para el navegador."""
    with open(MODELO_PKL, "rb") as f:
        datos = pickle.load(f)

    export_data = {
        "personas": datos["personas"],
        "umbral_desconocido": round(float(datos["umbral_desconocido"]), 4),
        "accuracy_cv": round(float(datos["accuracy_cv"]), 4),
        "estadisticas": datos["estadisticas"],
        "centroides": {
            p: [round(float(x), 5) for x in emb.tolist()]
            for p, emb in datos["centroides"].items()
        },
        "banco_por_persona": {
            p: [
                [round(float(x), 4) for x in fila.tolist()]
                for fila in embs
            ]
            for p, embs in datos["banco_por_persona"].items()
        },
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(export_data, f, separators=(",", ":"))

    tam_kb = os.path.getsize(JSON_OUT) / 1024.0
    print(f"Exportado exitosamente: {JSON_OUT} ({tam_kb:.1f} KB)")


if __name__ == "__main__":
    exportar()
