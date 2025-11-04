import pandas as pd
from preprocessing import preprocess  # This module is local
import os
import logging
import json
import joblib


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global xform_params

    # determine model path
    model_dir = os.getenv("AZUREML_MODEL_DIR")      # env variable AZUREML_MODEL_DIR holds the path to the model folder (injected by AzureML)
    subdir = os.listdir(model_dir)[0]               # for custom_model output, all files are located inside a subdirectory

    model_path = os.path.join(model_dir, subdir, "model.pkl")
    xform_params_path = os.path.join(model_dir, subdir, "xform_params.json")

    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

    with open(xform_params_path, 'r') as f:
        xform_params = dict(json.load(f))

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info(f"request received: {raw_data}")
    data = json.loads(raw_data)["data"]
    df = pd.DataFrame(data)
    df, _ = preprocess(df, xform_params)
    predictions = model.predict(df)

    logging.info("Request processed")

    return predictions.tolist()