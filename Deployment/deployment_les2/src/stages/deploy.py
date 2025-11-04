import os
import json
import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Model

# Parse the name of the endpoint to create
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--endpoint_name", type=str)
parser.add_argument("--example_payload", type=str)     # For pedagogical reasons only
args = parser.parse_args()


# AzureML injects DEFAULT_IDENTITY_CLIENT_ID, which contains the ClientId of the Managed Identity assigned to the cluster.
# This must be copied into AZURE_CLIENT_ID so DefaultAzureCredential picks it up.
os.environ["AZURE_CLIENT_ID"] = os.environ["DEFAULT_IDENTITY_CLIENT_ID"]

# Initialize MLClient using AzureML-injected environment variables
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
    resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
    workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
)


# Define the endpoint (the logical service (HTTP) endpoint)
endpoint = ManagedOnlineEndpoint(
    name=args.endpoint_name,
    auth_mode="key"  # Allows authentication using token/key — convenient for learning, insecure in production
)

ml_client.begin_create_or_update(endpoint).result()

# Define the deployment (actual model + resources)
deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint.name,
    model=ml_client.models.get(name=args.model_name, label="latest"),  # Fetches the latest version of the model. Always pin versions in production, this is for demonstration purposes only.
    environment="azureml://registries/azureml/environments/sklearn-1.5/versions/26",
    code_configuration=CodeConfiguration(
        code=".",   #  Will upload all files/dirs in root (the stages directory from our notebook perspective). This enables preprocessing.py to be available for import to score.py
        scoring_script="score.py"
    ),
    instance_type="Standard_D2as_v4", # Avoids DS-family quota limits
    instance_count=1
)

# Create or update both endpoint and deployment
ml_client.begin_create_or_update(deployment).result()

# Assign 100% traffic to the 'default' deployment, can only be done after the creation of the Deployment
endpoint.traffic = {"default": 100}
ml_client.begin_create_or_update(endpoint).result()

# Create an example datapoint to use for testing (for pedagogical reasons only)
example_payload = {
    "data": [
        {
            "LotFrontage": 65.0,
            "GrLivArea": 1710.0,
            "GarageArea": 548.0,
            "Neighborhood": "CollgCr",
            "HouseStyle": "2Story",
            "ExterQual": "Gd",
            "MasVnrType": "Stone",
            "YearBuilt": 2003,
            "YrSold": 2010,
            "OverallQual": 7
        }
    ]
}

with open(args.example_payload, "w") as text_file:
    text_file.write(json.dumps(example_payload))