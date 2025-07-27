from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[feature]) for feature in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]
    
    prediction = model.predict([values])[0]
    result = "🩺 You may have diabetes." if prediction == 1 else "✅ You do not have diabetes."

    return render_template('index.html', prediction_text=result)

if __name__ == '_main_':
    app.run(debug=True)