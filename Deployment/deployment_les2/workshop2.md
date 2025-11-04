# Workshop 2: Reproduceerbare Workflows met AzureML Pipelines

---

# Het Doel
In deze workshop maken we een volledige pipeline waarbij we beginnen met de data en eindigen met een gedeployed model. In deze pipeline wordt elke stap apart opgezet waardoor we herbruikbare onderdelen krijgen. Als we later een nieuw model willen deployen hoeven we maar een aantal stappen aan te passen en kunnen we zo de hele pipeline opnieuw runnen.

---


# 1. Hoe Ziet de Pipeline Eruit?
We maken een volledige ML pipeline in AzureML. Deze bestaat uit verschillende stappen. Elk van deze stappen is herbruikbaar en heeft hoogstens wat specifieke aanpassingen nodig per model. Deze pipeline bestaat uit de volgende stappen:

1. **Split**: Split de data in een train-test set.
2. **preprocess**: Clean de data en maak nieuwe features indien van toepassing
3. **Train**: Train een model met de gecleande data.
4. **Evaluate**: Evalueer de prestaties van het model.
5. **Tag**: Geef de prestaties mee aan het Model Asset.
6. **Deploy**: Deploy het model naar een Endpoint.
7. **Test**: Test de endpoint.

Deze pipeline kan vervolgens worden uitgevoerd met maar 1 commando. Dit in tegenstelling tot wat we in de vorige les gedaan hebben, waarbij je iedere stap nog los moest runnen.  


## 1.1 Pipelines en Dataflow
Een pipeline is in makkelijke termen een trechter die je data vanzelf de juiste richting in stuurt, vandaar wordt dit in zijn geheel ook wel een **Dataflow Orchestrator** genoemd. Deze pipeline bestaat uit verschillende **nodes** (ookwel **components**), dit zijn de onderdelen zoals *training* en *evalutation* etc. De verbindingen tussen die **nodes** zijn de **edges**. Deze representeren waar de data naartoe gaat. 

Door deze verbindingen weet AzureML welke stappen afhankelijk zijn van anderen, wat het dus ook mogelijk maakt wanneer je iets wil testen, je niet alles opnieuw hoeft uit te voeren. De outputs per **component** kunnen namelijk worden onthouden zodat je altijd de output van de ene **component** kan gebruiken om de volgende **component** te testen, zonder dat je alle code telkens opnieuw uit hoeft te voeren. 

Een **component** in AzureML is een soort doosje. Deze heeft een in- en een output. In het midden zit je python bestand die de stap ook daadwerkelijk uitvoert. Je kunt dus heel makkelijk bepalen wanneer je data naar welke **component** gaat.


---


# 2. Stap 1: Een Compute Cluster klaarzetten voor de Pipeline
In de vorige workshop hadden we een **Compute Instance** gebruikt voor het runnen van onze code. Dit keer gebruiken we een **Compute Cluster**, deze zijn handig voor automatische taken die schaalbaar moeten zijn. Een cluster kan namelijk zelf meerdere resources gebruiken als er veel tegelijk gedaan wordt, en deze ook weer afschalen als er weinig runt. 


## 2.1 Een Compute Cluster maken (ClickOps)

1. Ga naar [Azure ML Studio](https://ml.azure.com/) and selecteer jouw workspace.
2. In Het linker menu, klik op **Manage** > **Compute** > **Compute clusters**.
3. Klik **+ New** om een nieuwe cluster te maken. 
4. In het **Virtual Machine** tabje:
   - **Virtual Machine Size**: Selecteer **Standard_DS3_v2** 
   - Rest laten zoals het is.
5. In het **Advanced Settings** tabje:
   - **Compute name**: Vul de naam in: `mlops-cluster` (Zorg dat je deze naam kiest, die komt terug in de code die je in de volgende stappen gaat gebruiken.)
   - **Minimum number of nodes**: `0`
   - **Maximum number of nodes**: `1` (we hebben er ook maar 1 nodig)
   - **Idle seconds before scale down**: Zet deze op `120` (opstarten kan even duren)
   - Rest laten zoals het is.
6. Klik **Create** om de cluster te maken.

> **Workspace/resource group verwijderd?:** Als je de vorige les je workspace/resource group hebt verwijderd moet je deze eerst nog opnieuw maken. Doe het via [Azure ML Studio](https://ml.azure.com/) en volg de stappen. Als je vast loopt kun je altijd nog even naar de uitleg in les 1 kijken.

## 2.2 Een Managed Identity maken
Nu moeten we nog ervoor zorgen dat de cluster niet aan je persoonlijke account gebonden is, daarom maken we een **Managed Identity** aan. Met zo een Identity kunnen ook andere mensen de cluster gebruiken.

1. Ga naar de [Azure Portal](https://portal.azure.com).
2. Bovenaan in de zoekbalk, zoek: **Managed Identities**.
3. Klik **+ Create**.
4. onder het **Basics** tabje:
   - Selecteer jouw **Subscription**
   - Kies jouw **Resource Group**
   - Kies een logische naam zoals: `mlops-id`
   - Kies dezelfde  **Regio** als jouw AzureML Workspace (West Europe).
5. Klik **Review + Create**, dan **Create**.


## 2.3 De juiste rollen aan de Managed Identity geven
Om de Managed Identity de juiste rechten te geven:

1. In de Azure Portal, ga naar jouw **AzureML Workspace**. (Een manier om daar te komen is door op de starpagina op je subscriptie te klikken, dan Klik je op Resources en vervolgens op de naam van je workspace.)
2. Klik **Access Control (IAM)** > **+ Add**. **Add role assignment**.
3. Onder het **Role** tabje, kies de volgende rollen (omdat je maar 1 rol per keer kan selecteren moet je stap 3, 4 en 5 twee keer herhalen voor beide rollen):
   - ✅ **AzureML Data Scientist**
   - ✅ **AzureML Compute Operator**
   >Deze stap is wat raar omdat het niet zo goed toont welke je hebt geselecteerd in het rollen tabje, om te kijken of je de rol goed hebt geselecteerd kun je in het bovenste veld onder het members tabje zien welke rol je gaat toewijzen.
4. Onder **Members**:
   - Verander "Assign access to" naar `Managed identity`
   - Klik **+ Select members**:
      - Kies jouw Subscription
      - Bij **Managed Identity** selecteer `User-assigned managed identity`
      - Kies de Managed Identity die je net hebt gemaakt (`mlops-id`)
      - Klik **Select**
5. Klik **Review + Assign**

> - Met de **Data Scientist** rol kun je modellen trainen en registreren.  
> - De **Compute Operator** role is nodig om modellen te deployen en endpoints te maken.  


# 2.4 De Managed Identity aan de Compute Cluster koppelen

1. In Azure AI Studio, ga naar **Manage > Compute > Compute clusters**.
2. Klik op je cluster (`mlops-cluster`).
3. Ga naar het **Details** tabje.
4. Zoek **Managed identity** en klik op het ✏️ (edit) icoontje.
5. Kies op deze pagina:
    - Klik op **Assign a managed identity**
    - houdt **identity type** op _User-assigned_
    - In het zoekveld kies Managed Identity (`mlops-id`)
6. Klik **Update** om ze te koppelen aan elkaar.

Nu heeft je cluster de juiste rechten om automatische workloads aan te pakken.


---

# 3. Stap 2: Een notebook maken voor Pipeline Development
In deze stap maken we een **Compute Instance** aan om de code die we gaan gebruiken in de pipeline te schrijven en te testen. We hebben dus een compute cluster voor het runnen van de pipeline, en een compute instance voor het runnen van code die de pipeline opzet.


## 3.1 Een Compute Instance maken

1. In [Azure AI Studio](https://ml.azure.com/), selecteer je workspace.
2. In het linker menu, ga naar **Manage > Compute > Compute Instances tab**.
3. Klik **+ New** en vul in:
   - **Name**: Kies een logische naam, zoals: `workshop2-<jouw_naam>`
   - **Virtual Machine size**: Selecteer **Standard_DS11_v2**  
   - Onder **Advanced Settings**, zet de **Auto-shutdown** aan na **60** minuten. (of iets minder om zuiniger te zijn)
4. Klik **Create**. Dit kan enkele minuten duren.


## 3.2 Een nieuwe notebook maken
Zodra je compute klaar is voor gebruik en aan staat:

1. In het linker menu, selecteer **Notebooks**.
2. Klik **+files > Create new file**
3. Noem het bestand `pipeline-create.ipynb` en sla het op de default locatie op
4. bovenaan de notebook interface:
   - Selecteer je nieuwe **Compute Instance** uit de dropdown
   - Kies `Python 3.10 - SDK v2`


## 3.3 Initialiseer de SDK Client
In deze stap verbind je met AzureML, later hier meer over. Vul voor nu de eerste cel van je notebook in:

```py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()  # Portable, works both in interactive and automated workloads
ml_client = MLClient.from_config(credential=credential)  # Load config from current workspace (only works on personal compute)
```

Een AzureML SDK client is een Python-object uit de Azure Machine Learning SDK (Software Development Kit) dat je gebruikt om te communiceren met je Azure Machine Learning workspace.

Met die client kun je vanuit code (bijvoorbeeld een Jupyter notebook, VS Code, of een script) direct werken met AzureML-resources, in plaats van steeds via de webportal te klikken.

> ✅ Als deze runt zonder errors ben je goed verbonden en kun je verder gaan. 


## 3.4 Authenticatie met `DefaultAzureCredential`

De Azure SDK’s gebruiken de klasse `DefaultAzureCredential` om veilig in te loggen. Je hoeft hierdoor geen geheimen (zoals wachtwoorden of keys) in je code te zetten of handmatig in te stellen. In plaats daarvan probeert `DefaultAzureCredential` automatisch verschillende methoden in een vaste volgorde:

1. **Managed Identity** (voor geautomatiseerde workloads op Azure compute resources)  
2. **Omgevingsvariabelen**  
3. **Azure CLI login**  
4. **Visual Studio Code / Azure AI Studio SSO login** (voor interactieve sessies)  

Omdat je dit notebook draait op een persoonlijke compute instance binnen **Azure AI Studio**, herkent `DefaultAzureCredential` automatisch je login en logt in via je **gebruikersaccount** met **Single Sign-On (SSO)**.

Later in de workshop gaat je pipeline draaien als een geautomatiseerde workload op een compute cluster. Daar schakelt de authenticatie automatisch over naar de **managed identity** van het cluster — zonder dat je je code hoeft aan te passen.


## 3.5 Het gebruik van `MLClient`
De klasse `MLClient` is de belangrijkste interface om met AzureML te werken. Hiermee kun je onder andere jobs, datasets, modellen en pipelines beheren.

Als je `MLClient.from_config()` gebruikt, leest deze methode de workspace-metadata (zoals subscription ID, resource group en workspace naam) uit een lokaal `config.json` bestand.  Dit bestand wordt automatisch toegevoegd op **persoonlijke compute instances** binnen AzureML.  

Dit werkt alleen voor **interactieve workloads**. Bij **geautomatiseerde workloads** (zoals jobs op een cluster) is er geen lokaal config bestand aanwezig. In dat geval moet je de `MLClient` zelf aanmaken door de werkruimtegegevens expliciet mee te geven. Hoe je dat doet, laten we later in de workshop zien.


---


# 4. Stap 3: Registreer de Dataset via de SDK
Voordat we de onderdelen van de pipeline kunnen maken moeten we eerst onze data klaarzetten. Dit doen we iets anders dan de vorige workshop, dit keer maken we namelijk van de `.csv` een **Dataset asset**. 


## 4.1 Upload de Dataset (AmesHousing.csv)
Als je de data nog niet hebt geupload:

1. In Azure AI Studio, ga naar **Notebooks**.
2. Navigeer naar de folder met je pipeline notebook (die met je naam).
3. Klik op de 3 puntjes en klik op **Upload Files**.
4. Upload het `AmesHousing.csv` bestand.


## 4.2 Registreer het als een dataset asset
Voeg deze code toe aan je pipeline notebook en voer de code uit om het bestand te registreren:

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

dataset = Data(
    name="ames-housing-raw",
    description="Raw CSV of Ames Housing dataset",
    path="AmesHousing.csv",  # relative to notebook path
    type=AssetTypes.URI_FILE,
)

ml_client.data.create_or_update(dataset)
```
Hiermee upload je de data naar je AzureML Client, via deze client kun je nu altijd jouw data gebruiken.



## 4.3 Valideer de geregistreerde dataset in ML Studio
Om te kijken of alles is gelukt:

1. Ga naar [AzureML Studio](https://ml.azure.com/)
2. In het linker menu, **Assets** > **Data**
3. Als het goed is staat daar een dataset genaamd: `ames-housing-raw`
4. Je kunt erop klikken om alle metadata te zien van de dataset

Nu kun je de dataset gebruiken voor in je pipeline.


---


# 5. Pipeline - Splitting
Nu zijn we voorbereid voor het samenstellen van onze pipeline, hier gaan we stap voor stap de onderdelen van onze pipeline opstellen als **stages**. Elke keer wanneer we een stage van onze pipeline hebben gemaakt voegen we het component toe aan onze pipeline in het pipeline notebook. Het pipeline-notebook bevat dus alleen het toevoegen van de componenten, waarbij elk component refereert naar een stage script in de `stages/` folder.  

We beginnnen onze pipeline met het splitten van de data in een train-test set. Dit doen we als eerst om te zorgen dat we geen van de informatie van de train set in de testset terecht komt. 


## 5.1 Uitleg over nieuwe functies
Zoals eerder benoemd bevat elke node ook een python file, deze bewaren we in de `stages` folder. In de nodes gebruiken **command components**, deze kunnen worden gemaakt met de `command()` functie. Deze functie wordt gebruikt om de node op te zetten. We maken dus eerst het python script die de logica op de data uitvoerd, daarna kunnen we de logica omzetten naar een bruikbaar onderdeel voor onze pipeline met een ander- apart script. 

Daarnaast moeten we nog meer doen, we moeten namelijk in ons splitting script rekening houden met de in en output die ons script nodig heeft. Daarvoor gebruiken we een `ArgumentParser()`. Het is het makkelijkst om te denken dat de argumentparser wat plekjes vrijhoudt voor waar wij variabelen kunnen doorgeven aan de components. Dat kan beide voor de in- en uitvoer zijn voor ons script. Kijk nog even goed naar de code om te begrijpen hoe deze wordt gebruikt.

## 5.2 Het split script
Eerst maken we dus het script voor het splitten van de data `stages/split.py` (we doen ook hier meteen de feature selectie voor gemak):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# Parse command-line argument passed by AzureML to specify output location
parser = argparse.ArgumentParser()
parser.add_argument("--input_data")             # References a input URI
parser.add_argument("--test_size", type=float)
parser.add_argument("--random_state", type=int)
parser.add_argument("--X_train")                # References a output URI
parser.add_argument("--X_test")                 # References a output URI
parser.add_argument("--y_train")                # References a output URI
parser.add_argument("--y_test")                 # References a output URI
args = parser.parse_args()

# Read the dataset into a Dataframe.
df = pd.read_csv(args.input_data)

# Select features and target (same as Workshop 1)
features = [
    'LotFrontage', 'GrLivArea', 'GarageArea',
    'Neighborhood', 'HouseStyle', 'ExterQual', 'MasVnrType',
    'YearBuilt', 'YrSold', 'OverallQual'
]
target = 'SalePrice'

df = df.dropna(subset=[target])
X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

# Save outputs
X_train.to_csv(args.X_train, index=False)
X_test.to_csv(args.X_test, index=False)
y_train.to_csv(args.y_train, index=False)
y_test.to_csv(args.y_test, index=False)
```


## 5.3 Command component
In een nieuwe notebook cell van je pipeline notebook, maak de component aan:

```python
from azure.ai.ml import command, Input, Output, dsl

# Convenience, as we'll reuse it in many components.
SKLEARN_ENV = "azureml://registries/azureml/environments/sklearn-1.5/labels/latest"         # Curated Environment includes Scikit-learn for ML logic
ML_SDK_ENV = "azureml://registries/azureml/environments/python-sdk-v2/labels/latest"        # Curated Environment includes AzureML SDK for API access

split_component = command(
    code="./stages",                                # Directory hosting the component's code base. Becomes the working directory
    command="python split.py " \
                "--input_data ${{inputs.input_data}} " \
                "--test_size ${{inputs.test_size}} " \
                "--random_state ${{inputs.random_state}} " \
                "--X_train ${{outputs.X_train}} " \
                "--y_train ${{outputs.y_train}} " \
                "--X_test ${{outputs.X_test}} " \
                "--y_test ${{outputs.y_test}}",     # As this is a command components, we need to invoke the Python script
    environment=SKLEARN_ENV,                        # Needed, as Scikit learn functionality is used in the Python file
    display_name="split",
    inputs={                                        # Input slots
        "input_data": Input(type="uri_file"),
        "test_size": Input(type="string"),          # float-type is not yet supported by the SDK
        "random_state": Input(type="string")        # int-type is not yet supported by the SDK
    },
    outputs={                                       # Output slots
        "X_train": Output(type="uri_file"),
        "y_train": Output(type="uri_file"),
        "X_test" : Output(type="uri_file"),
        "y_test" : Output(type="uri_file")

    }
)
```
Zoals je ziet bij het maken van het `command`, kun je daar de variabelen meegeven. Dat kan door `${{}}` te gebruiken (AzureML DSL Syntax). 

Nu is het voor je gemaakt, maar wanneer je een keer zelf een pipeline gaat opstellen moet je zelf kijken wat je allemaal mee wil geven aan het volgende onderdeel van je pipeline. Zo meteen volgen veel soortgelijke stappen en dan kan wat herhaaldelijk overkomen, maar kijk nog even voor jezelf wat je allemaal mee moet geven per stap en wat elke stap doet. 


---


# 6. Pipeline - Prepping
De volgende stap is het preppen (preprocessen) van onze train en test sets.

## 6.1 Training-inference skew voorkomen
Dit bekent dat we willen voorkomen dat informatie vanuit onze train data naar onze testdata lekt, en andersom. Als we dat doen moeten we wel rekening houden met dat we precies dezelfde stappen uitvoeren op beide datasets. Als er een verschil inzit kan het namelijk invloed hebben op de evaluatie van het model, dan kunnen we dus niet meer met zekerheid zeggen hoe goed het model werkt. 

Om dat te voorkomen moeten we zorgen dat beide datasets door precies dezelfde functie gaan. In de volgende workshop gaan we het beter aanpakken en maken we zelfs gebruik van een sk-learn pipeline. 


## 6.2 Het preprocessing script
Het prep gedeelte is in twee stages opgedeeld, we hebben 1 preprocessing bestand, die de data op de juiste manier voorbereid en we hebben 1 bestand die de algemene logica afhandeld. 
Maak het python bestand `stages/preprocessing.py`:

```python
import pandas as pd


def get_transform_params(df: pd.DataFrame) -> dict:
    xform_params = {
        "lotfrontage_mean": df['LotFrontage'].mean(),
        "masvnrtype_mode": df['MasVnrType'].mode()[0],
        "categorical_cols": ['Neighborhood', 'HouseStyle', 'MasVnrType'],
        "exterqual_mapping": {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        "exterqual_na": 'TA'
    }
    return xform_params


def preprocess(df: pd.DataFrame, xform_params: dict=None) -> tuple[pd.DataFrame, dict]:

    if xform_params is None:
        params = get_transform_params(df)
    else:
        params = xform_params

    df = df.copy()  # Prevents mutating the original dataframe

    # Create/edit the features and fill the NaN's
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['OverallQual'] = df['OverallQual'].clip(lower=1, upper=10)
    df['LotFrontage'] = df['LotFrontage'].fillna(params['lotfrontage_mean'])
    df['MasVnrType'] = df['MasVnrType'].fillna(params['masvnrtype_mode'])
    df = pd.get_dummies(df, columns=params['categorical_cols'])
    df['ExterQual'] = df['ExterQual'].map(params['exterqual_mapping'])
    df['ExterQual'] = df['ExterQual'].fillna(params['exterqual_mapping'].get(params['exterqual_na']))

    # If the tranformation parameters were empty when calling the function, add the final columns to the params
    if not xform_params:
        params['final_columns'] = df.columns.tolist()
    else:
        df = df.reindex(columns=params['final_columns'], fill_value=0) # Deletes unspecified columns and fills missing ones with 0

    return df, params   # returns both the dataframe and the transformation parameters

```

Dit bestand bevat een processing functie die de nodige aanpassingen doet aan de data zodat hij klaar is voor gebruik in het model.

# 6.3 Modulair gebruik transformation parameters
In de vorige les hadden we onze transformation parameters gehardcode. Nu willen we dat voor veel mogelijk voorkomen, juist omdat we de onderdelen van onze pipeline op zo veel mogelijk plekken willen gebruiken. 

We houden dus onze transformation parameters bij door ze op te slaan in een dictionary. Als we die vervolgens bij onze bestanden opslaan kunnen we deze in de volgende stappen van onze pipeline makkelijk daar gebruiken. 

> **Transformation Parameters:** Dit zijn de waardes die we gebruiken om bijvoorbeeld de missende waardes te vullen met gemiddelden. Als het nog een beetje vaag overkomt, is het handig om nog even de code te bestuderen en te volgen hoe de transformation parameters worden aangemaakt. Kijk ook nog even waar ze laten dan weer worden toegepast, dit gebeurd in de training stap van de pipeline en in het deel waar we nieuwe voorspellingen gaan maken (de score/predict stap).  

## 6.4 Het prep script
Maak nu ook nog het script `stages/prep.py`:

```python
import pandas as pd
from preprocessing import preprocess
import json
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data")
parser.add_argument("--xform_params_in")
parser.add_argument("--prepped_data")
parser.add_argument("--xform_params_out")
args = parser.parse_args()

df = pd.read_csv(args.raw_data)

xform_params = None

# If theres added transformation parameters load them from the aguments, otherwise generate and save them.
if args.xform_params_in: 
    logging.info(f"Reading preprocessing params from {args.xform_params_in}")
    with open(args.xform_params_in, 'r') as f:
        xform_params = dict(json.load(f))

# Preprocess the df
df, xform_params = preprocess(df, xform_params)

# Save outputs
df.to_csv(args.prepped_data, index=False)

logging.info(f"Writing preprocessing params to {args.xform_params_out}")
with open(args.xform_params_out, "w") as f:
    json.dump(xform_params, f)
```

## 6.5 Het component maken
Maak het component in een nieuwe cell in je pipeline notebook:

```py
prep_component = command(
    code="./stages",    # Will also upload the preprocessing.py module
    command="python prep.py " \
                "--raw_data ${{inputs.raw_data}} " \
                "$[[--xform_params_in ${{inputs.xform_params_in}}]] "\
                "--prepped_data ${{outputs.prepped_data}} " \
                "--xform_params_out ${{outputs.xform_params_out}}", # $[[]] allows specifying optional parameters. If missing, the whole flag is ommited
    environment=SKLEARN_ENV,
    display_name="prep",
    inputs={
        "raw_data": Input(type="uri_file"),                         # Instead of hardcoding the input data, this allows parameterization for reusablilty
        "xform_params_in": Input(type="uri_file", optional=True)    # Preprocessing is not stateless, so we might need parameters derived during training
    },
    outputs={
        "prepped_data": Output(type="uri_file"),
        "xform_params_out": Output(type="uri_file")

    }
)
```


---


# 7. Pipeline - Training

Begin met het maken van je stage script `stages/train.py`

```python
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
```

In de laatste stap zie je dat we de transformation parameters opslaan samen met het model. Daarmee kunnen we het scoring script van de endpoint gebruiken zonder dat we de transformation parameters van een rare plek moeten halen, die staan namelijk op dezelfde locatie als het model.


## 7.1 Training component
In een nieuwe cell van je pipeline-notebook:

```python
model_asset_name = "minimal-model"  # Enforces consitent model asset name

train_component = command(
    code="./stages",
    command="python train.py " \
                "--X ${{inputs.X}} " \
                "--y ${{inputs.y}} " \
                "--xform_params ${{inputs.xform_params}} " \
                "--model_path ${{outputs.model_path}}",
    environment=SKLEARN_ENV,
    display_name="train-model",
    inputs={
        "X": Input(type="uri_file"),
        "y": Input(type="uri_file"),
        "xform_params": Input(type="uri_file"),
    },
    outputs={
        "model_path": Output(type="custom_model", mode="upload", name="minimal-model")  # custom_model output will automatically register the model as a directory
    }
)
```

De output van dit script is een model asset. 
De stap van het model registreren wordt op deze manier meteen voor ons gedaan. Deze stap zie je hier daarom niet meer terug. We kunnen nu dus direct door naar de deployment stap.


---


# 8. Pipeline - Deployment
Deze stap bevat beide het deployen én aanmaken van een endpoint. Dat is zo gedaan omdat wanneer je iets deployed op deze manier je meteen een endpoint mee moet geven. 

>**Note:** Belangrijk om te weten is, is dat we nu een pipeline maken waar alles instaat. Dit is alleen niet hoe het er meestal aan toe gaat. Vaak worden er twee pipelines gemaakt, een voor het maken van het model, en de andere voor het deployen naar een endpoint. Voor nu doen we dus alles in 1 pipeline omdat het makkelijker te volgen is en het beter het concept van een pipeline op

Maak het stage script aan voor de deployment stap `stages/deploy.py`:

```python
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
```

## 8.1 Deployment component
in een nieuwe cell van je pipeline-notebook:

```python
deploy_component = command(
    code="./stages",    # Will also include score.py, making it available for the deployment script
    command="python deploy.py " \
                "--model_name ${{inputs.model_name}} " \
                "--endpoint_name ${{inputs.endpoint_name}} " \
                "--example_payload ${{outputs.example_payload}}",
    environment=ML_SDK_ENV,
    display_name="deploy-model",
    inputs={
        "model_path": Input(type="custom_model"),       # Forces this step to wait for model registration, but is not used in the script
        "model_name": Input(type="string"),
        "endpoint_name": Input(type="string"),
    },
    outputs={
        "example_payload": Output(type="uri_file")      # Needed to create a data dependency (added for pedagogical reasons)
    }
)
```

# 8.2 Het score script
Voor de deployment hebben we ook nog het score script nodig. Deze zal de aanvragen van de voorspellingen die je maakt goed afhandelen. voeg `score.py` toe aan de stages:
```python
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
```


---


# 9. Pipeline - Testing 

Nu we een model hebben kunnen we ons model testen, maak hiervoor het stage script aan `stages/test.py`:

```python
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
```

## 9.1 Test component
In een nieuwe cel van onze pipeline-notebook:

```python
test_component = command(
    code="./stages",
    command="python test.py --endpoint_name ${{inputs.endpoint_name}} --example_payload ${{inputs.example_payload}}",
    environment="azureml://registries/azureml/environments/python-sdk-v2/versions/31",
    compute="mlops-cluster",
    display_name="test-endpoint",
    inputs={
        "endpoint_name": Input(type="string"),
        "example_payload": Input(type="uri_file"),    # Needed to create a data dependency (added for pedagogical reasons)

    }
)
```



## 9.2 🧱 Request Schema en Inference Format
AzureML verwacht een input in een bepaald formaat. Dit is afhankelijk hoe het model is getraind en hoe `score.py` eruit ziet.

In deze les hebben wij:
- Ons `score.py` script laadt een `model.pkl` in en voert dezelfde preprocessing logica uit als bij het trainen
- Het model verwacht een json payload (onze example data voor het uitvoeren van de voorspelling) in het formaat zoals hier onder: 

```json
{
  "data": [
    {
      "LotFrontage": 80.0,
      "GrLivArea": 1710,
      "GarageArea": 548,
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
```

Als de input niet overeen komt met het verwachtte formaat, zal de request falen. Dit is met een 400 error (bad request) of een interne error. Als je dit dus ziet heb je ergens een foutje gemaakt!


---


# 10. Pipeline - Evaluation
We gebruiken onze test set om de prestatie van het model in kaart te brengen.

Maak het script `stages/evaluate.py`:
```python
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
```


## 10.1 Evaluation component
In een nieuwe cell van je pipeline notebook:

```python
evaluate_component = command(
    code="./stages",
    command="python evaluate.py " \
                "--model_path ${{inputs.model_path}} " \
                "--X ${{inputs.X}} " \
                "--y ${{inputs.y}} " \
                "--metrics ${{outputs.metrics}}",
    environment=SKLEARN_ENV,
    display_name="evaluate model",
    inputs={
        "model_path": Input(type="custom_model"),
        "X": Input(type="uri_file"),
        "y": Input(type="uri_file")
    },
    outputs={
        "metrics": Output(type="uri_file")
    }
)
```


# 11. Pipeline - Tag
Als laatste stap voegen we de resultaten van de evaluation toe aan het model asset. Daarmee zijn de metrics zoals de loss gekoppeld aan het getrainde model. 

Maak hiervoor het laatste script `stages/tag.py`:

```python
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
```

## 11.1 Tag component
Deze voegen we ook toe aan ons pipeline-notebook in een nieuwe cell:

```python
tag_component = command(
    code="./stages",
    command="python tag.py " \
                "--model_name ${{inputs.model_name}} " \
                "--metrics ${{inputs.metrics}}",
    environment=ML_SDK_ENV,
    display_name="tag model",
    inputs={
        "model_name": Input(type="string"),
        "metrics": Input(type="uri_file")
    }
)
```


---


# 12. Het samenstellen van de pipeline
Eindelijk hebben we alle pipeline components klaar. Nu is het tijd om alles samen te voegen in de pipeline. 


## 12.1 De pipeline functie
Voor het maken van de pipeline maken we een functie die alle components aan elkaar plakt. Hiervoor gebruiken we de `@dsl.pipeline()` decorator. Hiermee weet AzureML dat de functie parameters moeten worden gezien als input/output objecten. 

```python
@dsl.pipeline()         # The job-name will default to the name of the function, experiment name will default to name of the directory.
def train_and_deploy(dataset_uri, model_name, endpoint_name):
    split = split_component(
        input_data=dataset_uri,
        test_size="0.2",              # Withhold 20% of the dataset as test set
        random_state="42"             # Fixing the random_state makes the process deterministic. ideal for reproducibility during development, but unsuitable for production
    )
    prep_train = prep_component(
        raw_data=split.outputs["X_train"]
    )
    prep_test = prep_component(
        raw_data=split.outputs["X_test"],
        xform_params_in=prep_train.outputs["xform_params_out"]
    )
    train = train_component(
        X=prep_train.outputs["prepped_data"],
        y=split.outputs["y_train"],
        xform_params=prep_train.outputs["xform_params_out"],
    )
    evaluate = evaluate_component(
        model_path=train.outputs["model_path"],
        X=prep_test.outputs["prepped_data"],
        y=split.outputs["y_test"],
    )
    tag = tag_component(
        model_name=model_name,
        metrics=evaluate.outputs["metrics"]
    )
    deploy = deploy_component( 
        model_path=train.outputs["model_path"],  # Needed to enforce dependency on train-component, but not actually used
        model_name=model_name,
        endpoint_name=endpoint_name
    )
    test = test_component(
        endpoint_name=endpoint_name,
        example_payload=deploy.outputs["example_payload"],
    )
```
> 💡 **Note:** Het `prep_component` is gebruikt beide voor de train- en test set. Dit komt doordat beide hetzelfde moeten worden voorbereid. De test set wacht nu op de transformation parameters van de train set, zodat deze set op precies dezelfde manier door de preprocessing komt. 

## 12.2 De pipeline submitten als job
Nu we de pipeline hebben samengesteld kunnen we het submitten als een **job**. Zodra we de code hiervoor uitvoeren zal hij meteen uitvoeren, voer het uit in een nieuwe cell van je pipeline-notebook:

```python
endpoint_name = "sklearn-endpoint"
ames_housing_data_asset = Input(type="uri_file", path="azureml:ames-housing-raw:1")

train_and_deploy_job = train_and_deploy(ames_housing_data_asset, model_asset_name, endpoint_name)
train_and_deploy_job.settings.default_compute = "mlops-cluster" # Specifies the default cluster for each component. (Can be overriden per component)

ml_client.jobs.create_or_update(train_and_deploy_job)
```

Klik vervolgens in de linker balk **Assets** > **jobs** om je actieve job te zien. Klik op de naam van de job om een diagram te tonen van de stappen van jouw pipeline. Als het fout gaat kun je dubbelklikken op de plek waar het fout is gegaan om de error in te zien. Het runnen van de job kan redelijk lang duren. ~10+ min, afhankelijk of je cluster is uitgevallen.

>⚠️ Houd er rekening mee dat argumenten die worden doorgegeven aan de decorator functie `Input`-objecten moeten zijn (of die converteerbaar zijn naar zo een object, bijv. `str`). Dit verklaart waarom we de Ames Housing Data Asset als een `Input`-object doorgeven en niet als een `str`: anders zou het geconverteerd worden naar een `str`-getypeerde `Input`, terwijl we een `uri_file`-getypeerde `Input` nodig hebben.  

🔁 AzureML cachet standaard de pipeline-uitvoer. Als invoerparameters of componentcode niet zijn gewijzigd, kunnen stappen worden overgeslagen om tijd te besparen. Om geforceerd opnieuw uit te voeren, wijzig je de inputs of stel je `.settings.force_rerun = True` in op de component of job.  


---


# 13. Controleer of alles gelukt is
Als je pipeline klaar is en je model is gedeployed naar een endpoint kunnen we alles testen. Het uiteindelijke doel blijft namelijk het aanroepen van het model via een endpoint. 


## 13.1 📂 Waar je de logs en outputs kunt vinden
Voor elke component, AzureML houdt automatisch bij:

| Wat                        | Waar                                      |
|-----------------------------|---------------------------------------------|
| **Logs** (stdout, stderr)   | Onder het **Outputs + logs** tabje               |
| **Artifacts** (e.g. model)  | Opgeslagen in de standaard AzureML **Datastore** |
| **Inputs/Outputs**          | Te zien in de diagram en component UI |
| **Environment snapshot**    | Zie **Environment & Inputs**       |

> 📁 Check **azureml-logs/** voor handige debug bestanden zoals:
> - `70_driver_log.txt`
> - `user_logs/stdout.txt`

## 13.2 Handmatig je endpoint testen
Als jouw `test.py` zonder errors is uitgevoerd is het waarschijnlijk goed gegaan. Echter wil je nog wel even kijken of het ook buiten AzureML goed werkt. Dat is namelijk wat de "klant" nodig zal hebben. Om te checken of alles werkt kunnen we net zoals in de vorige les een request sturen vanuit VSCode. 

Dit kun je doen door de code in test_endpoint.ipynb te runnen (zorg er net als in workshop 1 voor dat je de juiste endpoint_url en key invoert). Je kunt ook gebruik maken van de REST Client Extensie in VSCode, volg dan onderstaande stappen:

### Stap 1: Installeer REST Client Extensie in VSCode

1. Open **Visual Studio Code**  
2. Ga naar jouw **Extensions view**  
   `Ctrl+Shift+X` of `Cmd+Shift+X` op Mac  
3. Zoek voor:  
   **`REST Client`** van **Huachao Mao**  
4. Klik **Install**


### Stap 2: Jouw Endpoint URL en Access Key uit AzureML Studio halen

1. Visit [AzureML Studio](https://ai.azure.com/)  
2. In het linker menu, ga naar **Endpoints**  
3. Klik op je gedeployde **Online Endpoint**  
4. Selecteer het **Consume** tabje  
5. Kopieer de volgende dingen:
   - **Scoring URI**
   - **Access key** (Primary of Secondary)


### Step 3: Configureer Environment Variabelen in VSCode Settings

1. Open het **Command Palette** in VSCode  
   `Ctrl+Shift+P` of `Cmd+Shift+P` op Mac  
2. Typ en selecteer:  
   **Open User Settings (JSON)**  
3. In het `settings.json` bestand, update het volgende:

```json
{
  "rest-client.environmentVariables": {
    "$shared": {
      "endpoint_url": "https://<your-endpoint>.region.inference.ml.azure.com/score",
      "access_token": "your-access-key-here"
    }
  }
}
```

> Vervang de placeholders met jouw daadwerkelijke **endpoint URI** en **access key**.  

De `"rest-client.environmentVariables":` komt naast de andere instellingen te staan, hieronder een voorbeeld van een volledig `settings.json` bestand:

```json
// Neem dit niet letterlijk over!
{
    "C_Cpp.default.cppStandard": "c++17",
    "makefile.makePath": "C:\\msys64\\mingw64\\bin\\mingw32-make.exe",
    "extensions.ignoreRecommendations": true,
    "haskell.manageHLS": "PATH",
    "makefile.makefilePath": "C:\\msys64\\mingw64\\bin\\mingw32-make.exe",
    "github.copilot.nextEditSuggestions.enabled": true,
    "markdown-preview-enhanced.enablePreviewZenMode": true,
    "rest-client.environmentVariables": {
        "$shared": {
            "endpoint_url": "https://sklearn-endpoint.westeurope.inference.ml.azure.com/score",
            "access_token": "your-access-key-here"
        }
    }
}

```


### Stap 4: Het HTTP Request bestand maken

1. Maak een bestand in VSCode aan genaamd `call_endpoint.http`
2. Voeg het volgende toe aan het http bestand:

```http
### AzureML Online Endpoint Inference
POST {{endpoint_url}}
Authorization: Bearer {{access_token}}
Content-Type: application/json

{
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
```
3. Boven de **`POST`** Staat een knopje met **Send Request**.
4. Klik hierop om jouw endpoint te testen

Dit opent een nieuwe window, als het goed is staat hierin wat metadata met onderaan de voorspelling op de data die je hebt meegegeven:

```http
HTTP/1.1 200 OK
server: azureml-frontdoor
date: Sun, 31 Aug 2025 19:35:16 GMT
content-type: application/json
content-length: 11
x-ms-run-function-failed: False
x-ms-run-fn-exec-ms: 12.925
x-ms-server-version: azmlinfsrv/1.4.0
x-ms-request-id: 51d32cb5-d02c-4653-9a16-d9a7d2aec7ec
x-request-id: 51d32cb5-d02c-4653-9a16-d9a7d2aec7ec
azureml-model-deployment: default
azureml-model-session: default
connection: close

[
  212648.68
]
```

Nu heb je een werkende pipeline die in 1 keer jouw model online zet!


---


# 14. Samenvatting
Nu heb je een volledige pipeline online gezet met AzureML. Hiermee heb je:
- De nodige resources klaargezet, zoals de data en de compute instances 
- Alle pipeline componenten opgezet en aan elkaar verboden
- Een model getraind en gedeployed naar een online endpoint

## 14.1 Wat je er aan hebt
Met deze structuur kun je nu:

| Wat                        | Waarom het belangrijk is                                      |
|----------------------------|---------------------------------------------------------------|
| **Reproducibility**        | elke run is makkelijk reproduceerbaar met minimale moeite |
| **Traceability**           | Elke output is makkelijk terug te koppelen naar een job |
| **Automation**             | Pipelines kunnen handmatig, via een tijdsschema of via CI worden uitgevoerd |
| **Modularity**             | Componenten kunnen worden herbruikt in toekomstige pipelines |
| **Portability**            | Alles blijft werken, omdat het in de cloud wel gewoon werkt |



## 14.2 Voor de volgende les

In **Workshop 3**, focussen we op:

- 🧱 Het gebruik van **Scikit-learn Pipelines** om de preprocessing en model logica te bundelen in een enkel object
- 🧪 Het volledig vermijden van de **training–inference skew**, in plaats van er mee moeten dealen
- 🧾 We kijken naar **MLflow Tracking** om model runs te volgen en met elkaar te vergelijken


---

# Opdracht voor een deployment portfolio item
1. Stel je wil een nieuw model trainen op dezelfde dataset. Beschrijf voor ieder .py bestand in de stages map of deze aangepast moet worden of niet, en waarom.
2. Stel je wil een model trainen op een nieuwe dataset. Beschrijf opnieuw voor ieder .py bestand wat er aangepast moet worden en waarom.
3. Evalueer op basis van opdracht 1 en 2 wat het voordeel is van deze pipeline-aanpak ten opzichte van de aanpak die we in deployment_les1 gedaan hadden. Zijn er ook nadelen? Beschrijf ook wat je zelf als makkelijk en als lastig hebt ervaren.

**Extra:**
Voer de aanpassingen in de scripts uit en maak een werkende pipeline die een model traint op een andere dataset dan de Aimes Housing dataset. 

# Veel voorkomende errors:
1. **KeyError: 'DEFAULT_IDENTITY_CLIENT_ID'**
    - Check of je managed identity goed aan je cluster is gekoppeld.


