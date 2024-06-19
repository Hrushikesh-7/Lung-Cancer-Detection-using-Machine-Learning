from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

# Load the trained model
with open('lung_cancer_lr_unbalanced.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    gender = request.form['gender']
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    peer_pressure = int(request.form['peer_pressure'])
    chronic_disease = int(request.form['chronic_disease'])
    fatigue = int(request.form['fatigue'])
    allergy = int(request.form['allergy'])
    wheezing = int(request.form['wheezing'])
    alcohol_consuming = int(request.form['alcohol_consuming'])
    coughing = int(request.form['coughing'])
    shortness_of_breath = int(request.form['shortness_of_breath'])
    swallowing_difficulty = int(request.form['swallowing_difficulty'])
    chest_pain = int(request.form['chest_pain'])
    
    # Convert categorical inputs to numerical values
    gender = 1 if gender.upper() == 'M' else 0
    
    # Create input array for prediction
    input_features = np.array([gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
                               fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, 
                               swallowing_difficulty, chest_pain]).reshape(1, -1)
    
    # Make prediction using loaded model
    prediction = model.predict(input_features)[0]
    prediction_label = 'YES' if prediction == 1 else 'NO'
    # Return prediction result
    return redirect(url_for('result', prediction=prediction_label))

# Define route for result
@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction_text='Predicted Lung Cancer: {}'.format(prediction))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
