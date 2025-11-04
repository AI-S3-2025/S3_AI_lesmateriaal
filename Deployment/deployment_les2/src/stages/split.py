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