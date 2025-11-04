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