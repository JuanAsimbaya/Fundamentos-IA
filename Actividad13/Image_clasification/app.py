# pip install streamlit
# pip pip install joblib
# pip install scikit-learn

# Simple Streamlit web app for ML inference on the Iris dataset
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Cargar modelo entrenado
@st.cache_resource
def load_my_model():
    model = load_model("Actividad13/Image_clasification/iam_vs_famous_classification_model.keras")
    return model

model = load_my_model()

# Configuración de la página
st.set_page_config(page_title="Clasificador de Rostros", page_icon="🖼️")

st.title("Clasificador de Imágenes: Juan vs Otra Imagen")
st.write("Selecciona una imagen del listado y el modelo te dirá si corresponde a **Juan** o a **Otra Imagen**.")

# Mostrar foto de referencia fija
st.subheader("Foto de referencia de Juan")
st.image("imagenes/Juan_ref.jpeg", caption="Juan (referencia)", use_column_width=True)

# Listado de imágenes disponibles para prueba
imagenes_disponibles = [
    "imagenes/img1.jpg",
    "imagenes/img2.jpg",
    "imagenes/img3.jpg",
    "imagenes/img4.jpg",
    "imagenes/img5.jpg",
    "imagenes/img6.jpg",
    "imagenes/img7.jpg",
    "imagenes/img8.jpg",
    "imagenes/img9.jpg",
    "imagenes/img10.jpg",
    "imagenes/img11.jpg",
]

# Selector de imagen
opcion = st.selectbox("Elige una imagen para clasificar:", imagenes_disponibles)

if opcion:
    st.image(opcion, caption=f"Imagen seleccionada: {opcion}", use_column_width=True)

    # Preprocesar imagen
    img = image.load_img(opcion, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predicción
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]

    # Mostrar resultado
    if pred_class == 0:
        st.success("✅ La imagen corresponde a **Juan**")
    else:
        st.warning("⚠️ La imagen corresponde a **Otra Imagen**")
