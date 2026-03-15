import numpy as np 
import pandas as pd 
import os 

raw_csv_path = "/ibex/project/c2261/datasets/obelix/all.csv"
raw_cif_path = "/ibex/project/c2261/datasets/obelix/all_randomized_cifs/"

df = pd.read_csv(raw_csv_path)
# we drop some values that seems to be invalid:
df = df[df['Ionic conductivity (S cm-1)'] != '<1E-10']
df =df[df['Ionic conductivity (S cm-1)'] != '<1E-8']

assert np.sum(df['Ionic conductivity (S cm-1)'].astype(float)) == 0.543190439021392

features = df.loc[:, ["ID", 'Ionic conductivity (S cm-1)']]
features = features.drop_duplicates(inplace=False)
features['Ionic conductivity (S cm-1)'] = features['Ionic conductivity (S cm-1)'].astype(float)


assert len(features) == 562 

counter = 0
for cif_id in features["ID"]:
    cif_file = os.path.join(raw_cif_path, f"{cif_id}.cif")
    if not os.path.isfile(cif_file):
        print(f"missing CIF file: {cif_file}")
        features = features[features["ID"] != cif_id]
        counter += 1

print(f"removed {counter} entries due to missing CIF files")
assert len(features) == 562 - 277


features.to_csv("/ibex/project/c2261/datasets/obelix/obelix_cleaned.csv", index=False)
print("saved cleaned csv to /ibex/project/c2261/datasets/obelix/obelix_cleaned.csv")