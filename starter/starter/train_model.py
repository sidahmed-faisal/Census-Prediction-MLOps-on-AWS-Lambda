"""
Script to train machine learning model.
Author: Sidahmed Faisal
Data: Feb. 28th 2022
"""

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import logging, pickle , os , pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Initialize logging
logging.basicConfig(filename='training.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
logging.info("Started reading data")
file_path = "../data/census_cleaned.csv"
data = pd.read_csv(file_path)
logging.info(f"SUCCESS!: file was found in {file_path}")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=10,stratify=data['salary'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
logging.info("Started Trainning data")
lgbm_model = train_model(X_train, y_train)
logging.info(f"SUCCESS: Model Trainning Finsihes for {lgbm_model}")

# Evaluate the model
y_preds = inference(lgbm_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print("Precision: ", precision, " recall: ", recall, " fbeta: ", fbeta)

for value in data["education"].unique():
    print(value)

