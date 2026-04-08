import os
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, jsonify, request, render_template
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

model = load('models/modelofinal.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/predict', methods=['GET']) # /api/v1/predict?latitud=40.42&longitud=-3.70&superficie_m2=80&dormitorios=3
def predict_api():
    try:
        params = {
            'latitud': request.args.get('latitud', np.nan, type=float),
            'longitud': request.args.get('longitud', np.nan, type=float),
            'superficie_m2': request.args.get('superficie_m2', np.nan, type=float),
            'dormitorios': request.args.get('dormitorios', np.nan, type=float),
            'baños': request.args.get('banos', np.nan, type=float),

            # opcionales
            'floor': request.args.get('floor', None, type=str),
            'exterior': request.args.get('exterior', None, type=str),
            'conservation': request.args.get('conservation', None, type=str),
            'heating': request.args.get('heating', None, type=str),
            'air_conditioning': request.args.get('air_conditioning', None, type=str),
            'elevator': request.args.get('elevator', None, type=str),
            'build_year': request.args.get('build_year', None, type=float),

            # columnas raw que la pipeline puede esperar
            'features': None,
            'descripcion': None,
            'url': None,
            'precio': None,
            'media': None,
            'points_of_interest': None,
            'energy_data': None
        }

        input_data = pd.DataFrame([params])
        prediction = model.predict(input_data)[0]

        return jsonify({
            'precio_estimado_euros': round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        params = {
            'latitud': request.form.get('latitud', type=float),
            'longitud': request.form.get('longitud', type=float),
            'superficie_m2': request.form.get('superficie_m2', type=float),
            'dormitorios': request.form.get('dormitorios', type=float),
            'baños': request.form.get('banos', type=float),

            'floor': request.form.get('floor'),
            'exterior': request.form.get('exterior'),
            'conservation': request.form.get('conservation'),
            'heating': request.form.get('heating'),
            'air_conditioning': request.form.get('air_conditioning'),
            'elevator': request.form.get('elevator'),
            'build_year': request.form.get('build_year', type=float),

            'features': None,
            'descripcion': None,
            'url': None,
            'precio': None,
            'media': None,
            'points_of_interest': None,
            'energy_data': None
        }

        input_data = pd.DataFrame([params])
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction=round(float(prediction), 2),
            form_data=request.form
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f'Error en la predicción: {str(e)}',
            form_data=request.form
        )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)