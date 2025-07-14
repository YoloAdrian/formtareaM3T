from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos y transformadores
model   = joblib.load('titanic_modelnew.pkl')
scaler  = joblib.load('titanic_scalernew.pkl')
pca     = joblib.load('titanic_pcanew.pkl')
encoder = joblib.load('titanic_encodernew.pkl')

# Columnas en el mismo orden que el modelo espera
MODEL_COLS = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer inputs del formulario
        pclass = int(request.form['pclass'])
        sex    = request.form['sex']
        age    = float(request.form['age'])
        sibsp  = int(request.form['sibsp'])
        parch  = int(request.form['parch'])
        fare   = float(request.form['fare'])

        # 1) Codificar 'Sex'
        sex_df = pd.DataFrame([[sex]], columns=['Sex'])
        sex_df[['Sex']] = encoder.transform(sex_df[['Sex']])
        encoded_sex = sex_df['Sex'].iloc[0]

        # 2) Crear DataFrame con todas las columnas excepto PassengerId
        new_data = pd.DataFrame([{
            'Pclass': pclass,
            'Sex':    encoded_sex,
            'Age':    age,
            'SibSp':  sibsp,
            'Parch':  parch,
            'Fare':   fare
        }])

        # 3) Inyectar PassengerId dummy = 0
        new_data['PassengerId'] = 0

        # 4) Reordenar columnas
        new_data = new_data[MODEL_COLS]

        # 5) Escalar, PCA y predecir
        scaled = scaler.transform(new_data)
        pca_data = pca.transform(scaled)
        pred = model.predict(pca_data)

        # 6) Devolver respuesta
        return jsonify({'Supervivencia': 'Sí' if pred[0] == 1 else 'No'})

    except Exception as e:
        app.logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

