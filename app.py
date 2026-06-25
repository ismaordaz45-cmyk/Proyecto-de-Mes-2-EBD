from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# ─── Configurar el registro ───
logging.basicConfig(level=logging.DEBUG)

# ─── Cargar el pipeline entrenado (preprocesamiento + modelo) ───
model = joblib.load('modelo_final.pkl')
app.logger.debug('Pipeline cargado correctamente.')

# Opciones para los menús desplegables del formulario
NEIGHBOURHOOD_GROUPS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room']


@app.route('/')
def home():
    return render_template(
        'formulario.html',
        ng_options=NEIGHBOURHOOD_GROUPS,
        rt_options=ROOM_TYPES,
        prediction=None,
        error=None,
        form_values={}
    )


@app.route('/predict', methods=['POST'])
def predict():
    form_values = request.form.to_dict()
    try:
        # ─── Obtener y validar los datos enviados en el formulario ───
        neighbourhood_group = request.form.get('neighbourhood_group', '').strip()
        neighbourhood = request.form.get('neighbourhood', '').strip()
        room_type = request.form.get('room_type', '').strip()
        latitude_raw = request.form.get('latitude', '').strip()
        longitude_raw = request.form.get('longitude', '').strip()
        availability_raw = request.form.get('availability_365', '').strip()

        # Validaciones de campos obligatorios
        if not neighbourhood_group or neighbourhood_group not in NEIGHBOURHOOD_GROUPS:
            raise ValueError("Selecciona un distrito (neighbourhood_group) válido.")
        if not neighbourhood:
            raise ValueError("El barrio (neighbourhood) no puede estar vacío.")
        if not room_type or room_type not in ROOM_TYPES:
            raise ValueError("Selecciona un tipo de habitación válido.")
        if not latitude_raw or not longitude_raw:
            raise ValueError("Latitud y longitud son obligatorias.")
        if not availability_raw:
            raise ValueError("La disponibilidad (días/año) es obligatoria.")

        # Validaciones de tipo y rango (datos numéricos)
        try:
            latitude = float(latitude_raw)
            longitude = float(longitude_raw)
        except ValueError:
            raise ValueError("Latitud y longitud deben ser números (ej. 40.7128).")

        if not (40.4 <= latitude <= 41.0):
            raise ValueError("La latitud debe estar dentro del rango de NYC (40.4 a 41.0).")
        if not (-74.3 <= longitude <= -73.6):
            raise ValueError("La longitud debe estar dentro del rango de NYC (-74.3 a -73.6).")

        try:
            availability_365 = int(float(availability_raw))
        except ValueError:
            raise ValueError("La disponibilidad debe ser un número entero.")

        if not (0 <= availability_365 <= 365):
            raise ValueError("La disponibilidad debe estar entre 0 y 365 días.")

        # ─── Crear un DataFrame con los datos, en el mismo orden/columnas del entrenamiento ───
        data_df = pd.DataFrame([{
            'room_type': room_type,
            'neighbourhood': neighbourhood,
            'neighbourhood_group': neighbourhood_group,
            'latitude': latitude,
            'longitude': longitude,
            'availability_365': availability_365
        }])
        app.logger.debug(f'DataFrame creado: {data_df.to_dict(orient="records")}')

        # ─── Realizar la predicción (el pipeline aplica imputación, escalado y TargetEncoder) ───
        pred_log = model.predict(data_df)[0]
        # El modelo predice log1p(price); se revierte la transformación a USD
        pred_usd = float(np.expm1(pred_log))
        pred_usd = max(0.0, pred_usd)
        app.logger.debug(f'Predicción (log): {pred_log} -> USD: {pred_usd}')

        return render_template(
            'formulario.html',
            ng_options=NEIGHBOURHOOD_GROUPS,
            rt_options=ROOM_TYPES,
            prediction=round(pred_usd, 2),
            error=None,
            form_values=form_values
        )

    except ValueError as e:
        app.logger.error(f'Entrada inválida: {str(e)}')
        return render_template(
            'formulario.html',
            ng_options=NEIGHBOURHOOD_GROUPS,
            rt_options=ROOM_TYPES,
            prediction=None,
            error=str(e),
            form_values=form_values
        )
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return render_template(
            'formulario.html',
            ng_options=NEIGHBOURHOOD_GROUPS,
            rt_options=ROOM_TYPES,
            prediction=None,
            error='Ocurrió un error inesperado al procesar tu solicitud. Verifica los datos e intenta de nuevo.',
            form_values=form_values
        )


if __name__ == '__main__':
    app.run(debug=True)
