from flask import Flask, request
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from inmemorydb import InMemoryVectorDB

def create_app():
    # create and configure the app
    dataset = pd.read_csv('cancer.csv')
    cancer = dataset[dataset["diagnosis(1=m, 0=b)"] == 1].drop(columns=["diagnosis(1=m, 0=b)"]).values.tolist()
    non_cancer = dataset[dataset["diagnosis(1=m, 0=b)"] == 0].drop(columns=["diagnosis(1=m, 0=b)"]).values.tolist()

    app = Flask(__name__)
    db = InMemoryVectorDB()
    db.get_or_create_collection("cancer").add(cancer)
    db.get_or_create_collection("non_cancer").add(non_cancer)

    # a simple page that says hello
    @app.route('/cancer', methods=['POST'])
    def hello():
        request_data = request.json
        data = pd.DataFrame.from_records([request_data])
        model = load_model('cancer.h5')
        result = model.predict(data)
        related = db.get_or_create_collection("cancer").query(data.values.tolist()[0], 5)

        return {"result": {
            "related": related,
            "result": result
        }}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8080, debug=True)
