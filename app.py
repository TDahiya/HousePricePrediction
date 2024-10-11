from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load(open('best_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        area = float(request.form['area'])
        bhk = float(request.form['bhk'])
        bathroom = float(request.form['bathroom'])
        price_per_sqft = area / float(request.form['price'])  # Assuming price is passed to calculate price per sqft

        
        input_data = np.array([[area, bhk, bathroom, price_per_sqft]])
        
        
        input_data_scaled = scaler.transform(input_data)

        
        prediction = model.predict(input_data_scaled)

        return render_template('index.html', data=int(prediction[0]))

    return render_template('index.html', data=None)

if __name__ == '__main__':
    app.run(debug=True)
