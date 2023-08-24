import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
model_path = 'Model/model.pkl'

# Load the dataset and split into features and target
df = pd.read_csv('D:/Projects/Credit Card Default Prediction/notebook/data/Cleaned_Dataset.csv')
X = df.drop('default.payment.next.month', axis=1)
Y = df['default.payment.next.month']

# Train the model
model = RandomForestRegressor()
model.fit(X, Y)

# Save the trained model
joblib.dump(model, model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_name = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    df = pd.DataFrame([input_features], columns=features_name)
    prediction = model.predict(df)[0]

    if prediction == 1:
        res_val = "The Credit Card Holder is a DEFAULTER"
    else:
        res_val = "The Credit Card Holder is NOT a DEFAULTER"

    return render_template('index.html', prediction_text='Prediction: {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
