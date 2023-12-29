from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the saved model and preprocessor
loaded_model = joblib.load('medical_cost_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    sex = request.form['sex']
    smoker = request.form['smoker']
    region = request.form['region']

    # Create a new DataFrame for prediction
    new_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    })

    # Apply one-hot encoding to the new entry
    new_data_transformed = preprocessor.transform(new_data)

    # Make predictions
    predicted_charges = loaded_model.predict(new_data_transformed)
    result = f'Predicted Medical Cost: ${predicted_charges[0]:,.2f}'

    return render_template('index.html', result=result,age=age, bmi=bmi, children=children, sex=sex, smoker=smoker, region=region)

if __name__ == '__main__':
    app.run(debug=True)
