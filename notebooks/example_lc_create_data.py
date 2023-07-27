#%%
import os
import json
import pandas as pd

df = pd.read_csv("../data/raw/emails.csv", nrows=10)

# %%
for index, row in df.iterrows():
    with open(f"../data/processed/email_{index}.json", "w") as f:
        json.dump(row.to_dict(), f)

# %%
