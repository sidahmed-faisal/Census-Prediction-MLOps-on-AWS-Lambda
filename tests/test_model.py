from sklearn.exceptions import NotFittedError
from training import *
from training.ml import data, model
import pytest, os , pickle, logging
from sklearn.model_selection import train_test_split
import pandas as pd

"""
This class runs unit tests to make sure the model runs correct
"""

# Initialize logging
logging.basicConfig(filename='test.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
@pytest.fixture(scope="module")
def path():
    return "./data/census_cleaned.csv"

@pytest.fixture(scope="module")
def load_Data():
    # loads data to be used in the next tests
    data_path = "./data/census_cleaned.csv"
    logging.info(f"SUCCESS: file was found in {data_path}")
    data = pd.read_csv(data_path)
    return data

@pytest.fixture(scope="module")
def features():
    """
    Fixture to return the categorical features as argument
    """
    cat_features = [    "workclass",
                        "education",
                        "marital-status",
                        "occupation",
                        "relationship",
                        "race",
                        "sex",
                        "native-country"]
    return cat_features

@pytest.fixture(scope="module")
def train_dataset(load_Data, features):
    """
    Fixture to test the splitting and of the data
    """
    logging.info(f"started splitting the data")
    train, test = train_test_split( load_Data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=load_Data['salary']
                                )
        
    logging.info(f"SUCCESS: Data has been splitted")
    
    X_train, y_train, encoder, lb = data.process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train

def test_data_import(path):
    """
    Test data presence
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"SUCCESS: file was found in {path}")

    except FileNotFoundError as err:
        logging.error("FAILED: File not found")
        raise err
    
def test_model_path():
    """
    Check if the model is saved
    """
    model_path = "./model/trained_model.pkl"
    if os.path.isfile(model_path):
        try:
            _ = pickle.load(open(model_path, 'rb'))
        except Exception as err:
            logging.error(
            "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        print("some issue happend")

def test_model_predictions(train_dataset):
    """
    Check if model can run inference on splitted data
    """
    X_train, y_train = train_dataset
    model_path = "./model/model.pkl"
    model = pickle.load(open(model_path, 'rb'))

    try:
        model.predict(X_train)
    except NotFittedError as err:
        logging.error(f"Model is not fit, error {err}")
        raise err