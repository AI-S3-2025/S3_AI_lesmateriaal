import os
import json
import argparse
import requests
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint_name", type=str)
parser.add_argument("--example_payload", type=str)
args = parser.parse_args()

os.environ["AZURE_CLIENT_ID"] = os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)

# Retrieve the endpoint scoring URL and authentication key
endpoint = ml_client.online_endpoints.get(args.endpoint_name)
scoring_url = endpoint.scoring_uri

# Generate a token to authenticate with the endpoint
token = ml_client.online_endpoints.get_keys(name=args.endpoint_name).primary_key

# Load the example payload from input (added for pedagogical reasons only)
with open(args.example_payload, 'r') as text_file:
    json_example = text_file.read()

example_payload = json.loads(json_example)

# Define headers and sample input
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Send HTTP request
response = requests.post(scoring_url, headers=headers, json=example_payload)
print("Response:", response.text)