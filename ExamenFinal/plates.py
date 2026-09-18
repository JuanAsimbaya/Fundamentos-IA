import streamlit as st
import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr

# Cargar modelo YOLO entrenado
custom_model = YOLO('ExamenFinal/runs/detect/custom_yolo_model/weights/best.pt')

# Inicializar OCR
reader = easyocr.Reader(['en'], gpu=False)

# Función para extraer texto de la placa
def extract_plate_text(plate_crop):
    # Recorte inferior para evitar "ECUADOR" y "ANT"
    h = plate_crop.shape[0]
    plate_crop = plate_crop[int(h*0.35):, :]  # recorte inferior

    # Preprocesamiento
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    ocr_result = reader.readtext(thresh)

    # Filtrar por patrón de placa ecuatoriana (3 letras + 4 números)
    for _, text, _ in ocr_result:
        text = text.upper().replace(" ", "")
        if re.match(r'^[A-Z]{3}-?\d{4}$', text):
            return text
    return None

# Configuración de la página
st.set_page_config(page_title="Detector de Placas Ecuador", page_icon="🚗")
st.title("🚗 Detector y Reconocedor de Placas de Ecuador")

# Imagen de referencia
st.image("ExamenFinal/placa_referencia.jpg", caption="Ejemplo de placa ecuatoriana", width=150)

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen de un vehículo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", width=150)

    # Botón para detección con YOLO
    if st.button("Detectar Placa con YOLO"):
        # Streamlit pasa el archivo subido como un objeto tipo UploadedFile
        # La función predict() de Ultralytics YOLO espera una ruta de archivo o un array de imagen (NumPy)

        # Convertir el archivo subido a imagen NumPy
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Ejecutar predicción con YOLO
        results = custom_model.predict(img)

        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            # file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            plate_crop = img[int(y1):int(y2), int(x1):int(x2)]

            st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Placa detectada (recorte)")
            st.session_state["plate_crop"] = plate_crop
        else:
            st.warning("⚠️ No se detectó ninguna placa en la imagen.")

    # Botón para OCR
    if st.button("Reconocer Texto con OCR"):
        if "plate_crop" in st.session_state:
            st.image(cv2.cvtColor(st.session_state["plate_crop"], cv2.COLOR_BGR2RGB), caption="Recorte enviado a OCR")
            plate_text = extract_plate_text(st.session_state["plate_crop"])
            if plate_text:
                st.success(f"✅ Placa reconocida: **{plate_text}**")
            else:
                st.warning("⚠️ No se pudo leer la placa.")
        else:
            st.warning("⚠️ Primero detecta la placa con YOLO.")
