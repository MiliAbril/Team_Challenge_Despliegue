import os
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, jsonify, request, render_template
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


os.chdir(os.path.dirname(__file__))

# app = Flask(__name__)

# # Carga el modelo (guardado con joblib en el notebook)
# model = load('models/modelofinal.joblib')


# # ── Endpoint 1: Landing page ──────────────────────────────────────────────────
# @app.route('/', methods=['GET'])
# def hello():
#     return (
#         "<h1>API - Predicción de Precio de Vivienda en Madrid</h1>"
#         "<p>Endpoints disponibles:</p>"
#         "<ul>"
#         "  <li><code>GET /api/v1/predict</code> → predicción de precio</li>"
#         "  <li><code>GET /api/v1/retrain</code> → reentrenar con datos nuevos</li>"
#         "</ul>"
#     )


# @app.route('/api/v1/predict', methods=['GET'])
# def predict():

#     # Campos básicos
#     dormitorios  = request.args.get('dormitorios',  np.nan, type=float)
#     superficie   = request.args.get('superficie_m2', np.nan, type=float)
#     banos        = request.args.get('baños',         np.nan, type=float)
#     latitud      = request.args.get('latitud',       np.nan, type=float)
#     longitud     = request.args.get('longitud',      np.nan, type=float)

#     # Parámetros críticos
#     if any(isinstance(v, float) and np.isnan(v) for v in [latitud, longitud, superficie]):
#         return jsonify({'error': 'Parámetros obligatorios faltantes: latitud, longitud, superficie_m2'}), 400

#     # Reconstruimos el campo features como venía del CSV
#     features = {
#         'floor':                  request.args.get('floor',           None),
#         'elevator':               request.args.get('elevator',        None),
#         'air_conditioning':       request.args.get('air_conditioning', None),
#         'heating':                request.args.get('heating',         None),
#         'category':               request.args.get('category',        None),
#         'build_year':             request.args.get('build_year',      None, type=float),
#         'furnitured':             request.args.get('furnitured',      None),
#         'terraces':               request.args.get('terraces',        None),
#         'balconies':              request.args.get('balconies',       None),
#         'box':                    request.args.get('box',             None),
#         'car_places':             request.args.get('car_places',      None),
#         'garden':                 request.args.get('garden',          None),
#         'property_type':          request.args.get('property_type',   None),
#         'concierge':              request.args.get('concierge',       None),
#         'renovation_year':        request.args.get('renovation_year', None),
#         'floors':                 request.args.get('floors',          None),
#         'bedrooms':               request.args.get('bedrooms',        None),
#         'inside_renovation_year': request.args.get('inside_renovation_year', None),
#         'id':                     0,
#     }

#     # Construimos el DataFrame igual que el CSV raw
#     input_data = pd.DataFrame([{
#         'dormitorios':       dormitorios,
#         'superficie_m2':     superficie,
#         'baños':             banos,
#         'latitud':           latitud,
#         'longitud':          longitud,
#         'url':               '',
#         'descripcion':       '',
#         'precio':            '',
#         'media':             str([]),
#         'points_of_interest': str([]),
#         'energy_data':       str({}),
#         'features':          str(features),
#     }])

#     try:
#         prediction = model.predict(input_data)
#         return jsonify({'precio_estimado_euros': round(float(prediction[0]), 2)})
#     except Exception as e:
#         return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

# # ── Endpoint 3: Reentrenamiento ───────────────────────────────────────────────
# @app.route('/api/v1/retrain', methods=['GET'])
# def retrain():
#     global model
#     data_path = 'data/pisosmadrid_new.csv'

#     if not os.path.exists(data_path):
#         return "<h2>No se encontraron datos nuevos para reentrenar. No se hizo nada.</h2>", 404

#     try:
#         data = pd.read_csv(data_path, sep=';')

#         # Separar target (precio) de features
#         # El precio en el CSV original es un string tipo "{'price': '250.000', ...}"
#         # Aquí asumimos que en el CSV de reentrenamiento ya viene como número limpio
#         y = data['precio']
#         X = data.drop(columns=['precio'])

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.20, random_state=42
#         )

#         # Reentrenar el modelo (la pipeline completa)
#         model.fit(X_train, y_train)
#         mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

#         # Reentrenar con todos los datos
#         model.fit(X, y)

#         # Opcional: guardar el modelo reentrenado (descomentar si se desea persistencia)
#         # from joblib import dump
#         # dump(model, 'src/models/modelofinal.joblib', compress=3)

#         return jsonify({
#             'mensaje': 'Modelo reentrenado correctamente.',
#             'mape_validacion': round(mape, 4)
#         })

#     except Exception as e:
#         return jsonify({'error': f'Error durante el reentrenamiento: {str(e)}'}), 500


# if __name__ == '__main__':
#     app.run(debug=True)




app = Flask(__name__)

model = load('models/modelofinal.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/predict', methods=['GET'])
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