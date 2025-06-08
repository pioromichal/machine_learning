import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from ucimlrepo import fetch_ucirepo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

dataset = fetch_ucirepo(id=149)

df = dataset.data.features.copy()
df['Class'] = dataset.data.targets.values.ravel()

output_file = os.path.join(os.path.dirname(__file__), 'vehicles_data.csv')
df.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")
