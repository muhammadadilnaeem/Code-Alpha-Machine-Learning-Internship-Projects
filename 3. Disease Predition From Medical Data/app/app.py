from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open(r'E:\CodeAlpha Machine Learing 1 month\3. Disease Predition From Medical Data\jupyter notebook\02_heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            # Extract features from the form
            age = int(request.form['age'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            thalch = float(request.form['thalch'])
            oldpeak = float(request.form['oldpeak'])
            cp = request.form['cp']
            exang = int(request.form['exang'])
            slope = request.form['slope']
            thal = request.form['thal']

            # Mapping categorical variables to numerical
            cp_map = {'typical angina': 0, 'asymptomatic': 1, 'non-anginal': 2, 'atypical angina': 3}
            slope_map = {'downsloping': 0, 'flat': 1, 'upsloping': 2}
            thal_map = {'fixed defect': 0, 'normal': 1, 'reversable defect': 2}

            cp = cp_map[cp]
            slope = slope_map[slope]
            thal = thal_map[thal]

            # Create an array for prediction
            data = np.array([[age, trestbps, chol, thalch, oldpeak, cp, exang, slope, thal]])

            # Make a prediction
            prediction = model.predict(data)[0]

            # Redirect to the result page with prediction
            return redirect(url_for('result', prediction=prediction))

        except ValueError:
            return render_template('form.html', error_message="Invalid input! Please enter correct values.")

    return render_template('form.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    if prediction == '0':
        message = "No Heart Disease"
        image_file = 'No heart disease.PNG'
    else:
        message = f"Heart Disease Detected: {interpret_prediction(prediction)}"
        image_file = 'heart disease.PNG'

    return render_template('result.html', message=message, image_file=image_file)

def interpret_prediction(prediction):
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

if __name__ == '__main__':
    app.run(debug=True)
