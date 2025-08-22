
import pandas as pd
import os

# Get path of current file (so it works no matter where you run from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV path
csv_path = os.path.join(BASE_DIR, "creditcard.csv")

# Load dataset
data = pd.read_csv(csv_path)

print("Dataset loaded! Shape:", data.shape)
print(data.head())
