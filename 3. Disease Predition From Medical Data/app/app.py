
# Import libraries
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

# Initialize a Flask app
app = Flask(__name__)

# Load the pre-trained heart disease prediction model from a pickle file
with open(r'E:\CodeAlpha Machine Learing 1 month\3. Disease Predition From Medical Data\jupyter notebook\02_heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)  # Load the model into memory

# Define the home route which renders the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template

# Define the form route for heart disease input data
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':  # Check if the form is submitted via POST
        try:
            # Extract features from the submitted form data
            age = int(request.form['age'])  # Age input as integer
            trestbps = float(request.form['trestbps'])  # Resting blood pressure input as float
            chol = float(request.form['chol'])  # Cholesterol level input as float
            thalch = float(request.form['thalch'])  # Maximum heart rate achieved input as float
            oldpeak = float(request.form['oldpeak'])  # ST depression induced by exercise input as float
            cp = request.form['cp']  # Chest pain type as string
            exang = int(request.form['exang'])  # Exercise-induced angina as integer (0 or 1)
            slope = request.form['slope']  # Slope of the peak exercise ST segment
            thal = request.form['thal']  # Thalassemia type as string

            # Convert categorical string variables into numerical values using predefined mappings
            cp_map = {'typical angina': 0, 'asymptomatic': 1, 'non-anginal': 2, 'atypical angina': 3}
            slope_map = {'downsloping': 0, 'flat': 1, 'upsloping': 2}
            thal_map = {'fixed defect': 0, 'normal': 1, 'reversable defect': 2}

            # Apply mappings to convert user input into numerical format
            cp = cp_map[cp]  # Convert chest pain type
            slope = slope_map[slope]  # Convert slope type
            thal = thal_map[thal]  # Convert thalassemia type

            # Create an array with all input features for model prediction
            data = np.array([[age, trestbps, chol, thalch, oldpeak, cp, exang, slope, thal]])

            # Make the prediction using the loaded model
            prediction = model.predict(data)[0]  # Get the prediction result (0 or 1)

            # Redirect to the result page, passing the prediction result as a URL parameter
            return redirect(url_for('result', prediction=prediction))

        except ValueError:
            # Handle cases where input values are invalid (e.g., wrong format)
            return render_template('form.html', error_message="Invalid input! Please enter correct values.")

    # Render the form template (form.html) for GET request
    return render_template('form.html')

# Define the result route which displays the prediction result
@app.route('/result')
def result():
    # Get the prediction value from the URL parameter
    prediction = request.args.get('prediction')

    # Based on the prediction, define the message and image to display
    if prediction == '0':
        message = "No Heart Disease"  # No heart disease detected
        image_file = 'No heart disease.PNG'  # Display appropriate image
    else:
        message = f"Heart Disease Detected: {interpret_prediction(prediction)}"  # Heart disease detected
        image_file = 'heart disease.PNG'  # Display heart disease image

    # Render the result template (result.html) with the message and image
    return render_template('result.html', message=message, image_file=image_file)

# Helper function to interpret the heart disease prediction result into different stages
def interpret_prediction(prediction):
    # Return a message based on the prediction result (1-4 for different stages)
    if prediction == '1':
        return "Stage 1 (Mild Heart Disease)"
    elif prediction == '2':
        return "Stage 2 (Moderate Heart Disease)"
    elif prediction == '3':
        return "Stage 3 (Advanced Heart Disease)"
    elif prediction == '4':
        return "Stage 4 (Severe Heart Disease)"
    else:
        return "Unknown Stage"

# Main entry point for the application
if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask app in debug mode





# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Load the trained model
# with open(r'E:\CodeAlpha Machine Learing 1 month\3. Disease Predition From Medical Data\jupyter notebook\02_heart_disease_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/form', methods=['GET', 'POST'])
# def form():
#     if request.method == 'POST':
#         try:
#             # Extract features from the form
#             age = int(request.form['age'])
#             trestbps = float(request.form['trestbps'])
#             chol = float(request.form['chol'])
#             thalch = float(request.form['thalch'])
#             oldpeak = float(request.form['oldpeak'])
#             cp = request.form['cp']
#             exang = int(request.form['exang'])
#             slope = request.form['slope']
#             thal = request.form['thal']

#             # Mapping categorical variables to numerical
#             cp_map = {'typical angina': 0, 'asymptomatic': 1, 'non-anginal': 2, 'atypical angina': 3}
#             slope_map = {'downsloping': 0, 'flat': 1, 'upsloping': 2}
#             thal_map = {'fixed defect': 0, 'normal': 1, 'reversable defect': 2}

#             cp = cp_map[cp]
#             slope = slope_map[slope]
#             thal = thal_map[thal]

#             # Create an array for prediction
#             data = np.array([[age, trestbps, chol, thalch, oldpeak, cp, exang, slope, thal]])

#             # Make a prediction
#             prediction = model.predict(data)[0]

#             # Redirect to the result page with prediction
#             return redirect(url_for('result', prediction=prediction))

#         except ValueError:
#             return render_template('form.html', error_message="Invalid input! Please enter correct values.")

#     return render_template('form.html')

# @app.route('/result')
# def result():
#     prediction = request.args.get('prediction')
#     if prediction == '0':
#         message = "No Heart Disease"
#         image_file = 'No heart disease.PNG'
#     else:
#         message = f"Heart Disease Detected: {interpret_prediction(prediction)}"
#         image_file = 'heart disease.PNG'

#     return render_template('result.html', message=message, image_file=image_file)

# def interpret_prediction(prediction):
#     if prediction == '1':
#         return "Stage 1 (Mild Heart Disease)"
#     elif prediction == '2':
#         return "Stage 2 (Moderate Heart Disease)"
#     elif prediction == '3':
#         return "Stage 3 (Advanced Heart Disease)"
#     elif prediction == '4':
#         return "Stage 4 (Severe Heart Disease)"
#     else:
#         return "Unknown Stage"

# if __name__ == '__main__':
#     app.run(debug=True)
