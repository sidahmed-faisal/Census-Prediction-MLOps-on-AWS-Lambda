import json
import requests

# Define sample data
data = {"age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
        }

# Base Url to make predictions
url="https://zcodljp6wn4anehvint7oxvtda0qabvz.lambda-url.us-east-1.on.aws/predict"

response = requests.post(
    url, data=json.dumps(data))

print(response.status_code)
print(response.json())