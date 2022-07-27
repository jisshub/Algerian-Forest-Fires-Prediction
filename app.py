import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    data_t = [list(data.values())]
    output = model.predict(data_t)[0]
    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print(data)
    features = [np.array(data)]
    output = model.predict(features)[0]
    print(output)
    if output == 1:
        output = "Fire"
    elif output == 0:
        output = "No Fire"
    return render_template('index.html', prediction_text=f'Algerian forest will be under {output}')


if __name__ == '__main__':
    app.run(debug=True)
