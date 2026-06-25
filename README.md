# Predictor de Precio de Airbnb NYC — Despliegue con Flask y Render

## ⚠️ Importante: sobre `modelo_final.pkl`

Este paquete ya incluye un `modelo_final.pkl` **funcional** (probado: carga, predice y se
puede desplegar sin errores). Sin embargo, fue entrenado con una **muestra sintética**
de demostración, NO con el CSV completo de 48,895 filas, porque mis herramientas de
descarga de archivos tienen un límite de tamaño y no pude traer el dataset completo
a este chat.

**Para obtener el modelo real/definitivo (idéntico al de tu notebook):**

```bash
pip install -r requirements.txt
python train_model.py
```

Esto descarga el CSV completo desde la URL original y reentrana exactamente el mismo
pipeline (Random Forest, Escenario 1, TargetEncoder) que seleccionaste en tu notebook.
Tarda menos de un minuto. Al terminar, sobrescribe `modelo_final.pkl` con el modelo
correcto. Después de eso, sigue los pasos de despliegue normalmente.

---

## 1. Cómo se construyó la aplicación

- **Framework:** Flask (`app.py`), con una sola ruta `/` (formulario) y `/predict` (POST).
- **Pipeline:** un `sklearn.pipeline.Pipeline` que incluye:
  - Imputación: `SimpleImputer` (mediana en numéricas, moda en categóricas)
  - Codificación de categóricas: `TargetEncoder` (sustituye cada categoría por la
    media suavizada de `log(price)`)
  - Escalado: `StandardScaler` en las variables numéricas
  - Modelo: `RandomForestRegressor` (n_estimators=300, min_samples_leaf=4, max_features='sqrt')
- **Variable objetivo:** `log1p(price)`. Al predecir, se revierte con `expm1()` para
  mostrar el resultado en USD reales.
- **Variables de entrada (las 6 seleccionadas en el notebook):**
  `room_type`, `neighbourhood`, `neighbourhood_group`, `latitude`, `longitude`,
  `availability_365`. El usuario solo ingresa estos valores originales; el pipeline
  se encarga internamente de todas las transformaciones.

## 2. Cómo se guardó y cargó el pipeline

- Se guardó con `joblib.dump(pipeline, 'modelo_final.pkl')` (ver `train_model.py`).
- Se carga una sola vez al iniciar la app con `joblib.load('modelo_final.pkl')`
  (ver `app.py`), evitando cargarlo en cada solicitud.

## 3. Archivos necesarios para el despliegue

```
flask_airbnb/
├── app.py                  # Aplicación Flask
├── modelo_final.pkl        # Pipeline entrenado (preprocesamiento + modelo)
├── templates/
│   └── formulario.html     # Formulario web (HTML + CSS)
├── requirements.txt        # Dependencias
├── Procfile                # Comando de inicio para Render
├── .gitignore
├── train_model.py          # Script para regenerar el modelo con datos completos
└── README.md
```

## 4. Cómo probar localmente

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Mac/Linux

pip install -r requirements.txt
python train_model.py          # genera el modelo_final.pkl real (recomendado)
python app.py
```

Abre `http://127.0.0.1:5000/`, completa el formulario y verifica que aparezca el
precio estimado.

## 5. Subir a GitHub

```bash
git init
git add .
git commit -m "Despliegue modelo de regresión Airbnb NYC"
git remote add origin https://github.com/<TU_USUARIO>/<TU_REPO>.git
git push -u origin master
```

## 6. Configuración en Render

1. Crea cuenta en [render.com](https://render.com) y conecta tu repositorio de GitHub.
2. **New Web Service** → selecciona el repo.
3. Configura:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Click en **Create Web Service** y espera el deploy.
5. Copia la URL pública generada (ej. `https://tuapp.onrender.com`) y pruébala.

## 7. Pruebas realizadas

- ✅ Carga del formulario (`GET /`) — status 200
- ✅ Predicción con datos válidos (`POST /predict`) — devuelve precio en USD
- ✅ Validación de entradas inválidas (texto en campo numérico) — muestra mensaje
  de error sin romper la app
- ✅ Validación de rangos (latitud/longitud fuera de NYC, disponibilidad fuera de 0-365)

## 8. Dificultades encontradas y solución

- El dataset completo no se pudo descargar dentro del entorno de generación de este
  paquete (límite de tamaño de las herramientas usadas), por lo que se incluyó
  `train_model.py` para que el usuario final regenere el modelo real con el CSV
  completo en su propia máquina, donde sí hay acceso completo a internet.
- TargetEncoder requiere `y` durante el ajuste; esto se maneja automáticamente
  dentro del `Pipeline` de scikit-learn, que pasa `y` a cada paso que lo necesite.
- Las categorías no vistas en entrenamiento (ej. un barrio nuevo escrito por el
  usuario) son manejadas por `TargetEncoder`, que usa la media global como
  respaldo, evitando errores en producción.
