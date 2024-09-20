---

# **🧠 Digit Recognition with CNN & Streamlit**

This project is part of the **Congo Rise InfoTech Machine Learning Internship** and aims to build a digit recognition model using the famous **MNIST dataset**. The model is deployed via a Streamlit web app, allowing users to upload images and receive real-time digit predictions. 🚀

## **🎯 Objective**

The main objective of this project is to create a **Convolutional Neural Network (CNN)** that can classify 28x28 grayscale images of handwritten digits (0-9). The project also includes the development of a user-friendly **Streamlit web app** for easy image upload and prediction.

-----

https://github.com/user-attachments/assets/cb558782-8239-4b3a-8d13-aa095fa1ecaa


------



## **🛠️ Project Life Cycle**

1. **📂 Load the Dataset:**
   - Load and visualize the MNIST dataset.
   - Split the data into training and test sets.

2. **🎨 Normalize the Data:**
   - Normalize pixel values from **0-255** to **0-1**.
   - Reshape the data to fit CNN input dimensions (28x28x1).

3. **🏗️ Build the CNN Model:**
   - Design a CNN architecture using layers like `Conv2D`, `MaxPooling2D`, and `Dense`.
   - Use **ReLU** as the activation function for intermediate layers and **Softmax** for the output layer.

4. **🚀 Train the Model:**
   - Compile the model using the **Adam optimizer** and **sparse categorical cross-entropy** loss function.
   - Train the model on the training dataset and evaluate it on the test set.

5. **🔍 Predict and Visualize Results:**
   - Use the trained model to predict digits from test images.
   - Visualize actual vs. predicted labels and assess the performance.

6. **💾 Save the Model:**
   - Save the trained model as `mnist_cnn_model.h5` for future use.

7. **💻 Create a Streamlit Web App:**
   - Develop a Streamlit app that allows users to:
     - Upload images 📸.
     - Preprocess the images for prediction.
     - Display the predicted digit in real time.

## **🌟 Features**

- **Handwritten Digit Recognition**: The model accurately predicts digits from the MNIST dataset.
- **Interactive Streamlit App**: Upload images and get real-time predictions.
- **Model Saving**: The trained CNN model can be saved and reused without retraining.
- **Visualization**: View actual vs. predicted labels with graphical insights.

## **📦 Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/muhammadadilnaeem/Congo-Rise-InfoTech-Machine-Learning-Internship-Projects.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd 4.%20Digit%20Recognizer
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## **🖼️ App Interface**

The web app allows you to upload an image, and it will automatically preprocess and predict the digit. The interface is simple and user-friendly, with real-time results displayed alongside the image.

## 🧠 Model Architecture

- **Convolutional Layers**: For feature extraction.
- **MaxPooling Layers**: To reduce spatial dimensions.
- **Dense Layers**: For classification.
- **Activation**: ReLU and Softmax.
- **Optimizer**: Adam.
- **Loss Function**: Sparse Categorical Cross-Entropy.

## **📊 Results & Performance**

- Achieved **high accuracy** on the test set.
- Detailed visualizations of actual vs. predicted digits are available in the notebook.

## **🤖 Future Work**

- Implementing **Transfer Learning** for further performance improvement.
- Supporting **colored images** in addition to grayscale.

## **🤝 Contributing**

Feel free to contribute by forking the repository and submitting a pull request. Issues and feature requests are welcome! 🛠️


---

Happy coding! 🧑‍💻✨



