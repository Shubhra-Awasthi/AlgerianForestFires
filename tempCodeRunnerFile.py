from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Model or scaler file not found: {e}")

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Parse and validate inputs
            data = {
                'Temperature': request.form.get('Temperature'),
                'RH': request.form.get('RH'),
                'Ws': request.form.get('Ws'),
                'Rain': request.form.get('Rain'),
                'FFMC': request.form.get('FFMC'),
                'DMC': request.form.get('DMC'),
                'ISI': request.form.get('ISI'),
                'Classes': request.form.get('Classes'),
                'Region': request.form.get('Region'),
            }
            # Check for missing or invalid inputs
            if any(v is None or v.strip() == "" for v in data.values()):
                return render_template('home.html', error="All fields are required.")
            
            # Convert inputs to float
            inputs = [float(data[key]) for key in data]
            
            # Scale and predict
            new_data_scaled = standard_scaler_model.transform([inputs])
            result = ridge_model.predict(new_data_scaled)
            
            return render_template('home.html', results=result[0])
        
        except Exception as e:
            return render_template('home.html', error=f"Error during prediction: {str(e)}")
    
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
