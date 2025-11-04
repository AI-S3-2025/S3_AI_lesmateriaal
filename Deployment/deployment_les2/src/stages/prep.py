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