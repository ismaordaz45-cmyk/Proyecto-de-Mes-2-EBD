"""
train_model.py
----------------
Entrena el pipeline FINAL para el proyecto de regresión NYC Airbnb,
replicando EXACTAMENTE el mejor modelo seleccionado en el notebook:

    Escenario 1 (6 características seleccionadas) + Random Forest
    Preprocesamiento: SimpleImputer + StandardScaler (numéricas)
                       SimpleImputer + TargetEncoder (categóricas)
    Objetivo: log1p(price), con filtro price <= 500 USD

Uso:
    pip install -r requirements.txt
    python train_model.py

Esto descarga el CSV completo (48,895 filas) desde la URL del dataset,
entrena el pipeline con TODOS los datos disponibles y guarda:
    - modelo_final.pkl     (pipeline completo: preprocesamiento + modelo)
    - features_info.json   (metadatos de las variables de entrada)

Ejecuta esto en tu máquina (con internet) para obtener el modelo
EXACTO equivalente al de tu notebook. Luego solo coloca modelo_final.pkl
en esta misma carpeta antes de subir el proyecto a GitHub/Render.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

SEED = 42
URL = "https://raw.githubusercontent.com/pjournal/boun01g-data-mine-r-s/gh-pages/Assignment/AB_NYC_2019.csv"

# Las 6 características seleccionadas en la Fase 4 del notebook
SELECTED_FEATURES = ['room_type', 'neighbourhood', 'neighbourhood_group',
                      'latitude', 'longitude', 'availability_365']
NUM_COLS = ['latitude', 'longitude', 'availability_365']
CAT_COLS = ['room_type', 'neighbourhood', 'neighbourhood_group']


def main():
    print("Descargando dataset...")
    df_raw = pd.read_csv(URL)
    print(f"Dimensiones originales: {df_raw.shape}")

    # Fase 3: eliminar price == 0 y crear log_price
    df = df_raw[df_raw['price'] > 0].copy()

    # Fase 5: filtrar outliers extremos price > 500 USD (igual que el notebook)
    df_model = df[df['price'] <= 500].copy()
    df_model['log_price'] = np.log1p(df_model['price'])
    y_model = df_model['log_price'].values

    X_model = df_model[SELECTED_FEATURES].copy()
    print(f"Registros para entrenamiento final: {len(X_model)}")

    # Preprocesador (idéntico al notebook)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(smooth='auto', target_type='continuous', random_state=SEED))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, NUM_COLS),
        ('cat', cat_transformer, CAT_COLS)
    ])

    # Modelo final: Random Forest (mejor experimento del notebook)
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None,
        min_samples_leaf=4, max_features='sqrt',
        random_state=SEED, n_jobs=-1
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    print("Entrenando pipeline final con todos los datos disponibles...")
    pipeline.fit(X_model, y_model)

    joblib.dump(pipeline, 'modelo_final.pkl')
    print("Pipeline guardado en: modelo_final.pkl")

    info = {
        'input_features': SELECTED_FEATURES,
        'num_cols': NUM_COLS,
        'cat_cols': CAT_COLS,
        'escenario': 'Caracteristicas (Escenario 1)',
        'modelo': 'Random Forest',
        'price_filter': 'price <= 500 USD',
        'target_transform': 'log1p(price) -> expm1 al predecir'
    }
    with open('features_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("Metadatos guardados en: features_info.json")
    print("\nListo. Copia modelo_final.pkl a la carpeta del proyecto Flask y despliega.")


if __name__ == '__main__':
    main()
