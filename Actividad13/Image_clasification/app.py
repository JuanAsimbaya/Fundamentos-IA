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
st.image("Actividad13/Image_clasification/imagenes/juan.jpeg", caption="Juan (referencia)", width=300)

# Listado de imágenes disponibles para prueba
imagenes_disponibles = [
    "Actividad13/Image_clasification/imagenes/img1.jpg",
    "Actividad13/Image_clasification/imagenes/img2.jpg",
    "Actividad13/Image_clasification/imagenes/img3.jpg",
    "Actividad13/Image_clasification/imagenes/img4.jpg",
    "Actividad13/Image_clasification/imagenes/img5.jpg",
    "Actividad13/Image_clasification/imagenes/img6.jpg",
    "Actividad13/Image_clasification/imagenes/img7.jpg",
    "Actividad13/Image_clasification/imagenes/img8.jpg",
    "Actividad13/Image_clasification/imagenes/img9.jpg",
    "Actividad13/Image_clasification/imagenes/img10.jpg",
    "Actividad13/Image_clasification/imagenes/img11.jpg",
]

# Selector de imagen
opcion = st.selectbox("Elige una imagen para clasificar:", imagenes_disponibles)

if opcion:
    st.image(opcion, caption=f"Imagen seleccionada: {opcion}", width=300)

    # Preprocesar imagen
    img = image.load_img(opcion, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predicción
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]

    # Mostrar resultado
    if pred_class == 1:
        st.success("✅ La imagen corresponde a **Juan**")
    else:
        st.warning("⚠️ La imagen corresponde a **Otra Imagen**")
