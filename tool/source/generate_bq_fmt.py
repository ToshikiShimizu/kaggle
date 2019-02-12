import pandas as pd
import json
train = pd.read_csv('../input/train.csv',nrows=100000)
schema = []
for column in train.columns:
    d = {}
    d["name"] = column
    d["type"] = "string"
    d["mode"] = "nullable"
    schema.append(d)
with open('table_schema.json', 'w') as f:
    json.dump(schema, f)
