# for data manipulation
import pandas as pd
import os

# for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# huggingface
from huggingface_hub import HfApi

# HuggingFace authentication
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path
DATASET_PATH = "hf://datasets/agmsk86/Tourism_Package_Prediction_mohan/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop CustomerID (not useful for ML)
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns

label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Target column
target_col = "ProdTaken"

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to HuggingFace
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="agmsk86/Tourism_Package_Prediction_mohan",
        repo_type="dataset",
    )

print("Files uploaded successfully.")
