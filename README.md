# 🏠 Predictor de Precios de Vivienda

Aplicación web de Machine Learning para estimar el precio de venta de inmuebles a partir de sus características principales.

## 🌐 Demo en directo

👉 [Acceder a la aplicación](https://team-challenge-despliegue-hi65.onrender.com/)

---

## 📋 Descripción

Este proyecto implementa un modelo de predicción de precios de vivienda entrenado sobre datos reales del mercado inmobiliario. El usuario puede introducir las características de un inmueble y obtener una estimación del precio en euros de forma inmediata.

---

## 🚀 Funcionalidades

- Formulario web interactivo para introducir los datos del inmueble
- API REST para la integración con otras aplicaciones
- Predicción en tiempo real mediante un modelo de ML preentrenado
- Soporte para parámetros opcionales (ascensor, calefacción, año de construcción, etc.)

---

## 🔧 Tecnologías utilizadas

- **Python 3.10+**
- **Flask** — servidor web y API REST
- **scikit-learn** — pipeline de preprocesamiento y modelo
- **pandas / numpy** — manipulación de datos
- **joblib** — serialización del modelo

---

## 📡 API REST

Endpoint disponible para consultas programáticas:

GET /api/v1/predict

### Parámetros obligatorios

| Parámetro | Tipo | Descripción |
|---|---|---|
| latitud | float | Latitud de la ubicación |
| longitud | float | Longitud de la ubicación |
| superficie_m2 | float | Superficie en metros cuadrados |
| dormitorios | float | Número de dormitorios |
| baños | float | Número de baños |
| build_year | float | Año de construcción |

### Parámetros opcionales

floor, exterior, conservation, heating, air_conditioning, elevator

### Ejemplo de uso

GET https://team-challenge-despliegue-hi65.onrender.com/api/v1/predict?latitud=40.42&longitud=-3.70&superficie_m2=80&dormitorios=3

### Respuesta

{ "precio_estimado_euros": 325000.00 }

---

## ▶️ Ejecución en local

1. Clonar el repositorio
git clone https://github.com/MiliAbril/Team_Challenge_Despliegue

2. Instalar las dependencias
pip install -r requirements.txt

3. Arrancar la aplicación
python app.py

La aplicación estará disponible en http://127.0.0.1:5000

---

## 👥 Autores

* [Milagros Abril](https://github.com/MiliAbril)
* [Pedro Herrero Bas](https://github.com/PedroHBas)
* [Manuel Ros Martínez](https://github.com/mrosm-dev)
