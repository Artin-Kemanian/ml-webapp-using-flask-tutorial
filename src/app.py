from utils import db_connect
engine = db_connect()

# your code here
from flask import Flask, request, render_template
import pickle
import numpy as np


# Se crea una instancia de la aplicación Flask, donde: __name__ es equivalente a  "__main__" .
app = Flask(__name__)

# Cargar modelo ya entrenado.
with open('models/modelo_iris.pkl', 'rb') as f:
    model = pickle.load(f)

#En Flask, @app.route('/') asocia una URL con una función de Python. Es decir:

#Cuando el navegador accede a la URL raíz del servidor (/), ejecuta la función home().

@app.route('/')
def home():
    return render_template('index.html')


#Define una ruta accesible desde el navegador al enviar el formulario (action="/predict").

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    output = int(prediction[0])

    labels = ["Setosa", "Versicolor", "Virginica"]
    return render_template('index.html', prediction_text=f'La flor es: {labels[output]}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)