import pandas as pd
import os
import shutil
import joblib
import argparse
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--X")
parser.add_argument("--y")
parser.add_argument("--xform_params")
parser.add_argument("--model_path")
args = parser.parse_args()

X = pd.read_csv(args.X)
y = pd.read_csv(args.y)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Serialize the model to the output model location (a URI provided by AzureML)
os.makedirs(args.model_path, exist_ok=True)
joblib.dump(model, os.path.join(args.model_path, "model.pkl"))

# Copy transformation parameters to the output model location (a URI provided by AzureML)
shutil.copy2(args.xform_params, os.path.join(args.model_path, "xform_params.json"))