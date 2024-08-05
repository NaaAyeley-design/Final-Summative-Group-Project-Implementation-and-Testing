import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model (adjust the path as needed)
@st.cache_resource
def load_model():
    return load_model('')
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = img_to_array(image)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def main():

    st.title("Breast Cancer Classification App")

    model = tf.keras.models.load_model("weights-004-0.3355.keras")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)[0][0]

            # Interpret results
            if prediction > 0.8:
                label = "malignant"
                confidence = prediction * 100
            else:
                label = "benign"
                confidence = (1 - prediction) * 100

            # Display result
            st.write(f"The image is classified as {label} with {confidence:.2f}% confidence.")

if __name__ == "__main__":
    main()