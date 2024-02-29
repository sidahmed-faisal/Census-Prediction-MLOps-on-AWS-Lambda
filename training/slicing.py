import os, pickle
import pandas as pd
from typing import List
from ml.model import inference, compute_model_metrics
from ml.data import process_data

def compute_slices(model, encoder, lb, df, categorical_features,slice_features: List[str]):
    """
    function that outputs the performance of the model on slices
    of the data for value of a given categorical feature
     ------
    model: XGBClassifier()
        model trained
    df : pd.DataFrame
        Cleaned dataframe.
    categorical_features : list[str]
        List of the names of the categorical features.
    slice_features : list[str]
         Name of the feature/s used to make slices (categorical features)

    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    """
    performance_metric ={}
    for feature in slice_features:
        for value in df[feature].unique():
            X_slice = df[df[feature] == value]
            X_slice, y_slice, _, _ = process_data(
            X_slice, categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
            preds = inference(model, X_slice)
            print(
            f"shape of preds: {preds.shape} & shape of y_slice: {y_slice.shape}")
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            performance_metric[value] = {'Feature':feature,"Value":value,'Precision': precision,
                                'Recall': recall, 'Fbeta': fbeta}
    with open('slice_output.txt', 'w') as f:
        for key, value in performance_metric.items():
            f.write(f"{slice_features} = {key}: {value}")
            f.write("\n")
    return performance_metric

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
# Load Data
data = pd.read_csv("../data/census_cleaned.csv")

# Load model
model = pickle.load(open('../model/model.pkl', 'rb'))

# Load Encoder
encoder = pickle.load(open('../model/encoder.pkl', 'rb'))

# Load LabelBinarizer
lb = pickle.load(open('../model/lb.pkl', 'rb'))

# Compute performance slices on education categories
compute_slices(model=model, encoder=encoder, lb=lb, df=data, categorical_features= cat_features,slice_features=['education'])