from flask import Flask, request
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


def create_app():
    # create and configure the app
    app = Flask(__name__)

    # a simple page that says hello
    @app.route('/cancer', methods=['POST'])
    def hello():
        request_data = request.json
        data = pd.from_dict(request_data)
        model = load_model('cancer.h5')
        result = model.predict(data)
        return {"result": result}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)
