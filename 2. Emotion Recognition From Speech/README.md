

# **Emotion Recognition from Speech 🎤💬**

![task 2](https://github.com/user-attachments/assets/5e568b54-f82a-4ffe-90d1-886ae0dc5f01)


## **Overview**
This project aims to recognize emotions from speech using a Long Short Term Memory (LSTM) model. It was developed during my internship as a Machine Learning Engineer at CodeAlpha.

## **Demo**
Check out the demo video showcasing the web app's capabilities in recognizing emotions from audio files!


https://github.com/user-attachments/assets/37a8fd29-6b5a-4e83-8121-c76826e0c988



## **Features**
- Predicts emotions such as **Fear**, **Sadness**, and more from speech audio.
- User-friendly web interface for easy interaction.
- Built using advanced machine learning techniques for accurate predictions.

## **Project Lifecycle**
1. **Data Collection** 📊
   - Gathered diverse audio samples for training the model.

![DADADAD](https://github.com/user-attachments/assets/a79d6113-a975-4d5a-8f0b-f35c360c3319)


2. **Data Preprocessing** 🧹
   - Cleaned and prepared the audio data for analysis.

![SPECTROGRAM](https://github.com/user-attachments/assets/8264e14d-9de5-4aa8-b178-919eaeaca29e)



3. **Data Augmentation** 🔄
   - Enhanced the dataset by creating variations of existing audio samples.
  
![AUGMONTED](https://github.com/user-attachments/assets/1a922a3e-a4a1-474f-8a43-f1144610d8a5)


4. **Feature Extraction** 🔍
   - Identified and extracted key features from the audio files.
  

5. **Model Training** 🏋️‍♂️
   - Trained the LSTM model on the processed data.

![model training](https://github.com/user-attachments/assets/a8a5131a-7812-4f8a-9203-de8772447584)


6. **Model Evaluation** 📈
   - Assessed the model's performance and improved accuracy.
![LSTM](https://github.com/user-attachments/assets/9e7d373c-9f86-4c2e-91e9-5f0c01ca553d)


7. **Save the Model** 💾
   - Saved the trained model for future predictions.

![model](https://github.com/user-attachments/assets/f7743b4b-11a3-4adf-813a-b00265ed82ef)


9. **Model Deployment** 🚀
   - Deployed the model as a web application for user access.

![Capture](https://github.com/user-attachments/assets/a6452e2e-b8b5-4a5a-969e-16a2310e0747)


## **Getting Started**
To run this project locally, follow these steps:

### **Prerequisites**
- Python 3.x
- Required libraries (listed in `requirements.txt`)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-recognition-speech.git
   cd emotion-recognition-speech
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web app:
   ```bash
   python app.py
   ```

## **📁 Repository Structure**
```
emotion-recognition-speech/
│
├── app/
│   └── app.py                # Main application file
├── docs/
│   ├── codealpha-emotion-recognition-from-speech.pdf
    ├── Emotion-Recognition-From-Speech.pdf             # Document for steps taken to make this project
├── jupyter notebook/
│   └──  codealpha-emotion-recognition-from-speech.ipynbb       # To Do Experimentation with data
├── LSTM model/
│   └── lstm_model.h5              # LSTM Model
├── test audio files/
│   ├── OAF_base_sad.wav
    ├── YAF_back_fear.wav             # Audio used for model prediction
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
└── ...
```


## **Acknowledgments**
Thanks to #CodeAlpha for the opportunity to work on this exciting project!


