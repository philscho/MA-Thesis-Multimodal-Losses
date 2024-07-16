import pandas as pd
import json

print("Loading json ...")
# Schritt 1: JSON-Datei laden und IDs extrahieren
json_file_path = '/home/data/mmssl/CC3m/CC3M_LLaVA.json'

# with open(json_file_path, 'r') as file:
    # json_data = json.load(file)
json_data = pd.read_json(json_file_path)

print("Extracting IDs ...")
# Angenommen, die Bilddateinamen enthalten die IDs, extrahieren wir sie
# z.B., wenn die Dateinamen im Format "image_123.jpg" vorliegen
# ids_from_json = [filename.split('_')[2] for filename in json_data['id']]
ids_from_json = [filename.split('_')[2] for filename in json_data['id']]
#print(ids_from_json[])

print("Loading parquet ...")
# Schritt 2: Parquet-Datei laden
parquet_file_path = '/home/data/mmssl/CC3m/mapper.parquet'
df = pd.read_parquet(parquet_file_path)

print("Filter parquet by ids ...")
# Schritt 3: Parquet-Daten filtern basierend auf den extrahierten IDs
filtered_df = df[df['key'].isin(ids_from_json)]

print("Writing to new parquet file ...")
# Schritt 4: Gefilterte Daten in eine neue Parquet-Datei speichern
output_parquet_file_path = '/home/data/CC3M_LLaVA/mapper_LLaVA.parquet'
filtered_df.to_parquet(output_parquet_file_path)

print("Done!")
