import argparse
import os
import pandas as pd
import joblib
import json
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
parser.add_argument("--X")
parser.add_argument("--y")
parser.add_argument("--metrics")
args = parser.parse_args()

# Load model
model_path = os.path.join(args.model_path, "model.pkl")
model = joblib.load(model_path)

# Load test set
X = pd.read_csv(args.X)
y = pd.read_csv(args.y)

# Predict
y_pred = model.predict(X)

# Evaluate
metrics = {
    "r2": r2_score(y, y_pred),
    "rmse": root_mean_squared_error(y, y_pred),
    "mea": mean_absolute_error(y, y_pred)
}

with open(args.metrics, "w") as f:
    json.dump(metrics, f)