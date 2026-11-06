"""
================================================================================
ARCHIVO DE INTERFAZ GRÁFICA PRINCIPAL: app.py
================================================================================
OBJETIVO DEL ARCHIVO:
Este es el archivo ejecutable principal de la aplicación de escritorio en Python.
Conecta la interfaz gráfica (Tkinter) con el motor de Inteligencia Artificial
definido en 'reconocimiento.py'.

FUNCIONES PRINCIPALES:
  1. Carga en segundo plano el modelo entrenado ('modelo_rostros.pkl') al abrir la ventana.
  2. Permite al usuario subir cualquier imagen desde su PC ('📂 Subir Imagen para Reconocer').
  3. Muestra la foto con el recuadro dibujado sobre el rostro (verde o rojo).
  4. Muestra el nombre de la persona identificada (ANTHONY, CESIA, ELVIS, THOMAS)
     o indica "❌ ROSTRO DESCONOCIDO" si se sube la foto de una 5ta persona.
  5. Grafica las barras de porcentaje de similitud con las 4 personas y permite
     ajustar en vivo el umbral de exigencia.
================================================================================
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk

# Importamos las funciones de entrenamiento y predicción desde nuestro motor de IA
from reconocimiento import (
    analizar_imagen,
    cargar_o_entrenar_modelo,
    entrenar_modelo,
)


class AplicacionReconocimientoFacial:
    """Clase principal que construye y controla la ventana del sistema de reconocimiento."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Reconocimiento Facial — ANTHONY | CESIA | ELVIS | THOMAS")
        self.root.geometry("1040x720")
        self.root.minsize(920, 640)
        self.root.configure(bg="#0f172a")

        # Variables de estado de la aplicación
        self.modelo_datos = None        # Guardará el modelo entrenado en memoria
        self.ruta_imagen_actual = None  # Ruta de la última foto subida por el usuario
        self.img_tk_actual = None       # Referencia de imagen para mostrarla en Tkinter

        # 1. Construir todos los botones, paneles y barras de la ventana
        self._construir_interfaz()

        # 2. Cargar el modelo entrenado apenas se abre la ventana (sin congelar la pantalla)
        self.root.after(150, self._cargar_modelo_inicial)

    # ==========================================================================
    # DISEÑO DE LA VENTANA (ENCABEZADO, VISOR DE FOTO Y PANEL DE RESULTADOS)
    # ==========================================================================
    def _construir_interfaz(self):
        # --- ENCABEZADO SUPERIOR ---
        header = tk.Frame(self.root, bg="#1e293b", height=74)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        titulo_box = tk.Frame(header, bg="#1e293b")
        titulo_box.pack(side=tk.LEFT, padx=24, pady=12)

        tk.Label(
            titulo_box,
            text="Sistema de Reconocimiento Facial",
            font=("Segoe UI", 16, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        ).pack(anchor="w")

        self.lbl_estado_modelo = tk.Label(
            titulo_box,
            text="Cargando modelo entrenado...",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        self.lbl_estado_modelo.pack(anchor="w")

        # --- BOTONES DE ACCIÓN (SUBIR IMAGEN Y RE-ENTRENAR) ---
        botones_box = tk.Frame(header, bg="#1e293b")
        botones_box.pack(side=tk.RIGHT, padx=24, pady=14)

        self.btn_subir = tk.Button(
            botones_box,
            text="📂  Subir Imagen para Reconocer",
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            cursor="hand2",
            command=self.seleccionar_imagen,
        )
        self.btn_subir.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_reentrenar = tk.Button(
            botones_box,
            text="🔄  Re-entrenar Modelo",
            font=("Segoe UI", 10),
            bg="#334155",
            fg="#e2e8f0",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            cursor="hand2",
            command=self.reentrenar_modelo_async,
        )
        self.btn_reentrenar.pack(side=tk.LEFT)

        # --- CONTENEDOR CENTRAL DIVIDIDO EN 2 COLUMNAS ---
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Columna Izquierda: Visor donde se muestra la fotografía analizada
        panel_izq = tk.Frame(
            main_frame,
            bg="#1e293b",
            highlightbackground="#334155",
            highlightthickness=1,
        )
        panel_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.lbl_imagen = tk.Label(
            panel_izq,
            text="Haz clic en «📂 Subir Imagen para Reconocer»\npara analizar una fotografía",
            font=("Segoe UI", 12),
            fg="#64748b",
            bg="#1e293b",
            justify=tk.CENTER,
        )
        self.lbl_imagen.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Columna Derecha: Panel de resultados y barras de similitud por persona
        panel_der = tk.Frame(
            main_frame,
            bg="#1e293b",
            width=370,
            highlightbackground="#334155",
            highlightthickness=1,
        )
        panel_der.pack(side=tk.RIGHT, fill=tk.Y)
        panel_der.pack_propagate(False)

        tk.Label(
            panel_der,
            text="RESULTADO DEL ANÁLISIS",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w", padx=20, pady=(20, 8))

        # Tarjeta donde aparece el nombre de la persona o "ROSTRO DESCONOCIDO"
        self.card_resultado = tk.Frame(panel_der, bg="#0f172a", padx=16, pady=18)
        self.card_resultado.pack(fill=tk.X, padx=20, pady=(0, 16))

        self.lbl_identidad_titulo = tk.Label(
            self.card_resultado,
            text="PERSONA IDENTIFICADA",
            font=("Segoe UI", 8, "bold"),
            fg="#64748b",
            bg="#0f172a",
        )
        self.lbl_identidad_titulo.pack(anchor="w")

        self.lbl_identidad_valor = tk.Label(
            self.card_resultado,
            text="Esperando imagen...",
            font=("Segoe UI", 18, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
            wraplength=300,
            justify=tk.LEFT,
        )
        self.lbl_identidad_valor.pack(anchor="w", pady=(6, 4))

        self.lbl_confianza_detalle = tk.Label(
            self.card_resultado,
            text="Sube cualquier foto para verificar a quién pertenece.",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0f172a",
            wraplength=300,
            justify=tk.LEFT,
        )
        self.lbl_confianza_detalle.pack(anchor="w")

        # --- BARRAS COMPARATIVAS PARA ANTHONY, CESIA, ELVIS Y THOMAS ---
        tk.Label(
            panel_der,
            text="SIMILITUD CON CADA PERSONA REGISTRADA",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w", padx=20, pady=(6, 10))

        self.frame_barras = tk.Frame(panel_der, bg="#1e293b")
        self.frame_barras.pack(fill=tk.X, padx=20)

        self.barras_personas = {}
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Similitud.Horizontal.TProgressbar",
            troughcolor="#0f172a",
            background="#3b82f6",
            thickness=10,
        )

        for persona in ["ANTHONY", "CESIA", "ELVIS", "THOMAS"]:
            fila = tk.Frame(self.frame_barras, bg="#1e293b")
            fila.pack(fill=tk.X, pady=6)

            top_fila = tk.Frame(fila, bg="#1e293b")
            top_fila.pack(fill=tk.X)

            lbl_p = tk.Label(
                top_fila,
                text=persona,
                font=("Segoe UI", 10, "bold"),
                fg="#e2e8f0",
                bg="#1e293b",
            )
            lbl_p.pack(side=tk.LEFT)

            lbl_pct = tk.Label(
                top_fila,
                text="0.0%",
                font=("Segoe UI", 9),
                fg="#94a3b8",
                bg="#1e293b",
            )
            lbl_pct.pack(side=tk.RIGHT)

            pb = ttk.Progressbar(
                fila,
                style="Similitud.Horizontal.TProgressbar",
                orient=tk.HORIZONTAL,
                length=100,
                mode="determinate",
                maximum=100.0,
            )
            pb.pack(fill=tk.X, pady=(4, 0))
            self.barras_personas[persona] = (pb, lbl_pct)

        # --- CONTROL DESLIZANTE (SLIDER) DEL UMBRAL DE ROSTRO DESCONOCIDO ---
        separador = tk.Frame(panel_der, bg="#334155", height=1)
        separador.pack(fill=tk.X, padx=20, pady=16)

        tk.Label(
            panel_der,
            text="UMBRAL DE ROSTRO DESCONOCIDO",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w", padx=20)

        self.var_umbral = tk.DoubleVar(value=42.0)
        self.lbl_umbral_info = tk.Label(
            panel_der,
            text="Similitud mínima requerida: 42.0%",
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#1e293b",
        )
        self.lbl_umbral_info.pack(anchor="w", padx=20, pady=(2, 4))

        self.slider_umbral = tk.Scale(
            panel_der,
            from_=35.0,
            to=65.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.var_umbral,
            bg="#1e293b",
            fg="#cbd5e1",
            highlightthickness=0,
            troughcolor="#0f172a",
            activebackground="#2563eb",
            showvalue=False,
            command=self._al_cambiar_umbral,
        )
        self.slider_umbral.pack(fill=tk.X, padx=20)

        tk.Label(
            panel_der,
            text="Si el rostro tiene una similitud menor a este umbral con las 4 personas, se clasifica como ROSTRO DESCONOCIDO.",
            font=("Segoe UI", 8),
            fg="#64748b",
            bg="#1e293b",
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=20, pady=(4, 0))

    # ==========================================================================
    # LÓGICA DE EVENTOS: CARGA DEL MODELO, SUBIDA DE IMAGEN Y RESULTADOS
    # ==========================================================================
    def _al_cambiar_umbral(self, valor):
        """Se ejecuta al mover la barra de umbral; recalcula en vivo si hay una foto cargada."""
        v = float(valor)
        self.lbl_umbral_info.configure(text=f"Similitud mínima requerida: {v:.1f}%")
        if self.ruta_imagen_actual and self.modelo_datos:
            self._ejecutar_analisis(self.ruta_imagen_actual)

    def _cargar_modelo_inicial(self):
        """Carga el modelo entrenado en un hilo secundario para mantener fluida la ventana."""
        def tarea():
            try:
                datos = cargar_o_entrenar_modelo(
                    callback_progreso=lambda idx, tot, p, arch: self.root.after(
                        0,
                        lambda: self.lbl_estado_modelo.configure(
                            text=f"Entrenando [{idx}/{tot}] — {p} ({arch})..."
                        ),
                    )
                )
                self.modelo_datos = datos
                self.root.after(0, self._actualizar_info_modelo)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Error al cargar el modelo", str(e)),
                )

        threading.Thread(target=tarea, daemon=True).start()

    def _actualizar_info_modelo(self):
        """Actualiza el indicador superior con la cantidad de fotos y la precisión del modelo."""
        if not self.modelo_datos:
            return
        est = self.modelo_datos["estadisticas"]
        total_caras = sum(v["detectados"] for v in est.values())
        acc = self.modelo_datos["accuracy_cv"] * 100.0
        umbral_calibrado = self.modelo_datos["umbral_desconocido"] * 100.0
        self.var_umbral.set(round(umbral_calibrado, 1))
        self.lbl_umbral_info.configure(text=f"Similitud mínima requerida: {umbral_calibrado:.1f}%")
        self.lbl_estado_modelo.configure(
            text=f"Modelo activo • 4 personas ({total_caras} fotos) • Precisión CV: {acc:.1f}%",
            fg="#4ade80",
        )

    def reentrenar_modelo_async(self):
        """Vuelve a analizar las 4 carpetas de fotos si se agregaron nuevas imágenes."""
        self.btn_reentrenar.configure(state=tk.DISABLED)
        self.btn_subir.configure(state=tk.DISABLED)

        def tarea():
            try:
                datos = entrenar_modelo(
                    callback_progreso=lambda idx, tot, p, arch: self.root.after(
                        0,
                        lambda: self.lbl_estado_modelo.configure(
                            text=f"Re-entrenando [{idx}/{tot}] — {p} ({arch})...",
                            fg="#fbbf24",
                        ),
                    )
                )
                self.modelo_datos = datos
                self.root.after(0, self._actualizar_info_modelo)
                if self.ruta_imagen_actual:
                    self.root.after(0, lambda: self._ejecutar_analisis(self.ruta_imagen_actual))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(
                    0,
                    lambda: (
                        self.btn_reentrenar.configure(state=tk.NORMAL),
                        self.btn_subir.configure(state=tk.NORMAL),
                    ),
                )

        threading.Thread(target=tarea, daemon=True).start()

    def seleccionar_imagen(self):
        """Abre el explorador de archivos de Windows para elegir la foto a reconocer."""
        if not self.modelo_datos:
            messagebox.showinfo("Espera un momento", "El modelo aún se está cargando.")
            return

        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen para reconocer rostro",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return

        self.ruta_imagen_actual = ruta
        self._ejecutar_analisis(ruta)

    def _ejecutar_analisis(self, ruta_imagen: str):
        """
        Envía la foto seleccionada a 'analizar_imagen()' (en reconocimiento.py)
        y actualiza la pantalla con el recuadro y los porcentajes de similitud.
        """
        umbral_actual = float(self.var_umbral.get()) / 100.0
        res = analizar_imagen(ruta_imagen, self.modelo_datos, umbral_personalizado=umbral_actual)

        # Convertir la imagen anotada de BGR (OpenCV) a RGB (Pillow/Tkinter) para mostrarla
        img_bgr = res["imagen_bgr"]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        ancho_disp = max(420, self.lbl_imagen.winfo_width() - 32)
        alto_disp = max(420, self.lbl_imagen.winfo_height() - 32)
        img_pil.thumbnail((ancho_disp, alto_disp), Image.Resampling.LANCZOS)

        self.img_tk_actual = ImageTk.PhotoImage(img_pil)
        self.lbl_imagen.configure(image=self.img_tk_actual, text="")

        # Si no se detectó ningún rostro en la imagen subida
        if not res["exito"] or len(res["resultados_rostros"]) == 0:
            self.lbl_identidad_valor.configure(text="SIN ROSTRO DETECTADO", fg="#fbbf24")
            self.lbl_confianza_detalle.configure(
                text="No se encontró ningún rostro claro en la fotografía subida."
            )
            for _, (pb, lbl_pct) in self.barras_personas.items():
                pb["value"] = 0.0
                lbl_pct.configure(text="0.0%")
            return

        # Tomar el resultado del rostro principal detectado
        principal = res["resultados_rostros"][0]
        nombre = principal["nombre"]
        es_conocido = principal["es_conocido"]
        similitud_pct = principal["similitud"] * 100.0
        mejor_candidato = principal["mejor_candidato"]

        # Mostrar en verde si es una de las 4 personas, o en rojo si es ROSTRO DESCONOCIDO
        if es_conocido:
            self.lbl_identidad_valor.configure(text=f"✅ {nombre}", fg="#4ade80")
            self.lbl_confianza_detalle.configure(
                text=f"Rostro reconocido con {similitud_pct:.1f}% de similitud biométrica."
            )
        else:
            self.lbl_identidad_valor.configure(text="❌ ROSTRO DESCONOCIDO", fg="#f87171")
            self.lbl_confianza_detalle.configure(
                text=(
                    f"No corresponde a ninguna de las 4 personas registradas.\n"
                    f"(Mayor coincidencia: {mejor_candidato} con {similitud_pct:.1f}%, "
                    f"debajo del umbral de {umbral_actual*100:.1f}%)."
                )
            )

        # Actualizar las 4 barras de progreso con el porcentaje de parecido de cada persona
        for persona, (pb, lbl_pct) in self.barras_personas.items():
            sim_p = principal["similitudes_por_persona"].get(persona, 0.0) * 100.0
            pb["value"] = min(100.0, max(0.0, sim_p))
            lbl_pct.configure(
                text=f"{sim_p:.1f}%",
                fg="#4ade80" if (es_conocido and persona == nombre) else "#94a3b8",
            )


# Punto de entrada al ejecutar: python app.py
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionReconocimientoFacial(root)
    root.mainloop()
