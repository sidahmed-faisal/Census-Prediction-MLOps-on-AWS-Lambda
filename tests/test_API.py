import os, json , logging
from fastapi.testclient import TestClient

from app import app

# Initialize logging
logging.basicConfig(filename='training.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Create Client to test the API
client = TestClient(app)

# Test index get endpoint
def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "App is running!"

# Test prediction endpoint
def test_predict_negative():

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
    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json()['prediction'] == "this person earns  ('<=50K',)"

# Test prediction endpoint
def test_predict_positive():

    data = {"age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    response = client.post("/predict", data=json.dumps(data))
    assert response.status_code == 200
    assert response.json()['prediction'] == "this person earns  >50K"

# Test invalid request body
def test_predict_invalid_request():
    data = {}
    response = client.post("/predict", json=json.dumps(data))
    assert response.status_code == 422
    logging.warning(f"The Request body has {len(data)} features you must provide 14 features")