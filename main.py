"""
API to serve the machine learning model predictions.
Author: Sidahmed Faisal
Data: Feb. 28th 2022
"""
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from training.ml.data import process_data
from training.ml.model import inference
import uvicorn
import pandas as pd
import numpy as np
import logging

# Initialize logging
logging.basicConfig(filename='training.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Initialize the App
app = FastAPI(title="Census Prediction API",
                description="This is an API to make inference on the machine learning model",
                version="1.0.0")

model_path = "./model/model.pkl"
lb_path = "./model/lb.pkl"
encoder_path = "./model/encoder.pkl"

model = pickle.load(open(model_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))

class InputData(BaseModel):
    # This example data is from the first row in the dataset
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')

@app.get('/')
async def index():
    return "App is running!"

@app.post('/predict')
async def prediction(census_data: InputData):
    data =  census_data.dict()
    age = data['age']
    workclass= data['workclass']
    fnlgt= data['fnlgt']
    education= data['education']
    education_num= data['education_num']
    marital_status= data['marital_status']
    occupation= data['occupation']
    relationship= data['relationship']
    race= data['race']
    sex= data['sex']
    capital_gain= data['capital_gain']
    capital_loss= data['capital_loss']
    hours_per_week= data['hours_per_week']
    native_country= data['native_country']
    # logging.info(f"{age}")
    
    # Convert the request to a dataframe
    inference_data = pd.DataFrame(data, index=[0])
    cat_features = [
                    "workclass",
                    "education",
                    "marital_status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native_country",
                    ]
    
    sample,_,_,_ = process_data(
                                inference_data, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

    prediction = model.predict(sample)

    # Return the prediction 
    if prediction>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K', 
    data['prediction'] ="this person earns "+ " "+ str(prediction)

    return data



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)