import argparse
import os
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

parser = argparse.ArgumentParser()
parser.add_argument("--model_name")
parser.add_argument("--metrics")
args = parser.parse_args()

os.environ["AZURE_CLIENT_ID"] = os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

# Load metrics
with open(args.metrics, 'r') as f:
    metrics = json.load(f)

latest_model_asset = ml_client.models.get(name=args.model_name, label="latest")

for key, value in metrics.items():
    latest_model_asset.tags[key] = str(value)

ml_client.models.create_or_update(latest_model_asset)