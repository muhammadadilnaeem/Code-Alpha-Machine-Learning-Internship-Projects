'''
This Script is Interface for MNIST Digit Recognition.
'''

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2

# Page configuration with relevant emoji
st.set_page_config(
    page_title="MNIST Digit Recognition 🧠",
    page_icon="🔢",
    layout="centered"
)

# Load your trained model
model = load_model(r'E:\Congo Rise Machine Learning Internship\4. Digit Recognizer\model\mnist_cnn_model.h5')

# Modernized CSS styling for UI/UX
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fd;
        }
        .header {
            color: #ffffff;
            text-align: center;
            font-family: 'Verdana';
            background-color: #cc75fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .upload-box {
            text-align: center;
            border: 2px dashed #5e60ce;
            padding: 25px;
            margin-top: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        }
        .stButton > button {
            background-color: #ff6347;
            color: #ffffff;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            display: block;
            text-align: center;
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #ff4500;
        }
        .result-box {
            border: 4px solid #3CB371;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
            font-size: 50px;
            font-weight: bold;
            color: #3CB371;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Display a heading with emoji
st.markdown('<div class="header"><h1>🧠 MNIST Digit Recognition 🔢</h1></div>', unsafe_allow_html=True)

# Upload multiple image formats (JPEG, PNG)
st.markdown('<div class="upload-box"><h3>📂 Upload a handwritten digit image (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # Preprocess the image for model input
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)

    # Normalize the image data
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)

    # Centered and styled predict button
    if st.button('🔍 Predict'):
        # Perform prediction
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)

        # Big, bold, and colored output with a bold square around it
        st.markdown(f"<div class='result-box'>🎯 Predicted Digit: {predicted_digit}</div>", unsafe_allow_html=True)




# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageOps
# import cv2

# # Page configuration with relevant emoji
# st.set_page_config(
#     page_title="MNIST Digit Recognition 🧠",
#     page_icon="🔢",
#     layout="centered"
# )

# # Load your trained model
# model = load_model('mnist_cnn_model.h5')

# # Modernized CSS styling for UI/UX
# st.markdown("""
#     <style>
#         .main {
#             background-color: #f8f9fd;
#         }
#         .header {
#             color: #ffffff;
#             text-align: center;
#             font-family: 'Verdana';
#             background-color: #5e60ce;
#             padding: 25px;
#             border-radius: 10px;
#             margin-bottom: 30px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
#         }
#         .upload-box {
#             text-align: center;
#             border: 2px dashed #5e60ce;
#             padding: 25px;
#             margin-top: 20px;
#             background-color: #ffffff;
#             border-radius: 10px;
#             box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
#         }
#         .predict-button {
#             background-color: #5e60ce;
#             color: #ffffff;
#             padding: 10px 15px;
#             border: none;
#             border-radius: 8px;
#             cursor: pointer;
#             font-size: 18px;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#             transition: background-color 0.3s ease;
#         }
#         .predict-button:hover {
#             background-color: #4a50b5;
#         }
#         .emoji {
#             font-size: 50px;
#             margin-bottom: 20px;
#         }
#         .result {
#             font-size: 24px;
#             color: #333333;
#             font-weight: bold;
#             text-align: center;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Display a heading with emoji
# st.markdown('<div class="header"><h1>🧠 MNIST Digit Recognition 🔢</h1></div>', unsafe_allow_html=True)

# # Upload multiple image formats (JPEG, PNG)
# st.markdown('<div class="upload-box"><h3>📂 Upload a handwritten digit image (JPEG, PNG, JPG)</h3></div>', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

#     # Preprocess the image for model input
#     image = ImageOps.grayscale(image)
#     image = image.resize((28, 28))
#     image = np.array(image)

#     # Normalize the image data
#     image = image / 255.0
#     image = image.reshape(1, 28, 28, 1)

#     # Predict button with emoji
#     if st.button('🔍 Predict'):
#         # Perform prediction
#         prediction = model.predict(image)
#         predicted_digit = np.argmax(prediction)

#         # Display prediction with an emoji
#         st.markdown(f"<div class='result'>🎯 Predicted Digit: {predicted_digit}</div>", unsafe_allow_html=True)

