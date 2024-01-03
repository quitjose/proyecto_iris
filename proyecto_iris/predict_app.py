import pickle
from flask import Flask, jsonify, request
from predict_service import predict_single

app = Flask('iris-predict')


modelos = {}

with open('models/iris-regresion_model.pck', 'rb') as f:
     modelos['regresion_model'] = pickle.load(f)

with open('models/iris-svm_model.pck', 'rb') as f:
    modelos['svm_model'] = pickle.load(f)

with open('models/iris-tree_model.pck', 'rb') as f:
    modelos['tree_model']  = pickle.load(f)

with open('models/iris-knn_model.pck', 'rb') as f:
    modelos['knn_model'] = pickle.load(f)


@app.route('/predict', methods=["POST"])
def predict():
    datos_json = request.get_json()

    modelo = datos_json.get('Modelo')
    petal_length = datos_json.get('PetalLengthCm')
    petal_width = datos_json.get('PetalWidthCm')

    iris, prediccion= predict_single(petal_length, petal_width, modelos[modelo])

    lista_python = prediccion.tolist()
    result = {
        'Modelo': modelo,
        'iris': iris,
        'prediccion': lista_python
    }

    return jsonify(result)
    

if __name__ == '__main__':
    app.run(debug=True, port=8000)




