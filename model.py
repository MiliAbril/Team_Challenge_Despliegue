import os
import ast
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from utils.maps import COLS_OPTIMIZE

from utils.utils import (aplanar_campos_anidados,crear_features_poi,limpiar_y_crear_features,drop_columns,)
from utils.kmeanscluster import KMeansCluster

os.chdir(os.path.dirname(__file__))

# ── 1. Carga de datos ─────────────────────────────────────────────────────────

df = pd.read_csv('./data/pisos_madrid.csv', sep='|')


# # ── 2. Extracción del target (precio) ─────────────────────────────────────────
# # precio es un diccionario string → extraemos la clave 'price' y limpiamos
# def extraer_precio(precio_str):
#     try:
#         d = ast.literal_eval(precio_str) if isinstance(precio_str, str) else precio_str
#         valor = d.get('price', None)
#         if valor is None:
#             return np.nan
#         return float(str(valor).replace('.', '').replace(',', '.').strip())
#     except:
#         return np.nan

# df['precio_final'] = df['precio'].apply(extraer_precio)
# df = df.dropna(subset=['precio_final'])

# y = df['precio_final'].values

# # Eliminamos columnas que no son features o que causan data leakage
# X = df.drop(columns=['precio_final', 'url', 'descripcion'])

# # ── 3. Train/Test split ───────────────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ── 2. Extracción del target (precio) ─────────────────────────────────────────
y = df['precio'].astype(str).str.extract(r'([\d\.]+) €').replace(r'\.', '', regex=True).astype(int).squeeze()

df = df.dropna(subset=['precio'])

X = df.drop(columns=['precio', 'url', 'descripcion'])

# ── 3. Train/Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ── 4. Pipeline de features (igual que en el notebook) ───────────────────────
cluster_transformer = ColumnTransformer(
    [
        ('cluster', KMeansCluster(n_clusters=6), ['latitud','longitud'])
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')

pipe_features = Pipeline(steps=[
    ('aplanar_campos',  FunctionTransformer(aplanar_campos_anidados)),
    ('poi_features',    FunctionTransformer(crear_features_poi)),
    ('final_clean',     FunctionTransformer(limpiar_y_crear_features)),
    ('cluster', cluster_transformer),
    ('drop_cols',       FunctionTransformer(drop_columns)),
])

# ── 5. Preprocesamiento numérico y categórico ─────────────────────────────────
preprocess = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ]), make_column_selector(dtype_include=np.number)),

    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ]), make_column_selector(dtype_include=object)),
])

# ── 6. Pipeline completa con XGBoost ─────────────────────────────────────────
# Hiperparámetros obtenidos con Optuna en el notebook
best_params_xgb = {
    'model__n_estimators':      964,
    'model__learning_rate':     0.042358543385845486,
    'model__max_depth':         5,
    'model__min_child_weight':  1,
    'model__subsample':         0.7875817229297305,
    'model__colsample_bytree':  0.916375202177841,
    'model__reg_alpha':         0.16164074832599548,
    'model__reg_lambda':        3.7736277459500225,}

columnas_a_quitar = COLS_OPTIMIZE

optimizacion = ColumnTransformer(
    [('optimizado', 'drop', columnas_a_quitar)],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')

xgb_base = XGBRegressor(random_state=42, objective='reg:absoluteerror', n_jobs=-1)

pipe_xgb = Pipeline(steps=[
    ('features',   pipe_features),
    ('optimizacion', optimizacion),
    ('preprocess', preprocess),
    ('model',      xgb_base),
])
pipe_xgb.set_params(**best_params_xgb)

# ── 7. Modelo final con transformación logarítmica del target ─────────────────
modelofinal = TransformedTargetRegressor(
    regressor=pipe_xgb,
    func=np.log1p,
    inverse_func=np.expm1,
)

# ── 8. Entrenamiento y evaluación ─────────────────────────────────────────────
modelofinal.fit(X_train, y_train)
mape = mean_absolute_percentage_error(y_test, modelofinal.predict(X_test))
print(f"MAPE en test: {mape:.4f}")

# Reentrenamiento con todos los datos antes de guardar
modelofinal.fit(X, y)

# ── 9. Guardado del modelo ────────────────────────────────────────────────────
os.makedirs('./models', exist_ok=True)
dump(modelofinal, './models/modelofinal.joblib', compress=3)
print("Modelo guardado en ./models/modelofinal.joblib")
