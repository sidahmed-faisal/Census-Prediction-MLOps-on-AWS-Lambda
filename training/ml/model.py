from sklearn.metrics import fbeta_score, precision_score, recall_score
# import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

from xgboost import XGBClassifier 

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define the hyperparameter distributions
    param_dist = {
        'max_depth': stats.randint(3, 10),
        'learning_rate': stats.uniform(0.01, 0.1),
        'subsample': stats.uniform(0.5, 0.5),
        'n_estimators':stats.randint(50, 200)
    }

    # Create the XGBoost model object
    xgb_model = XGBClassifier()

    # Create the RandomizedSearchCV object
    model = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')

    # Fit the GridSearchCV object to the training data
    model.fit(X_train, y_train)

    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", model.best_params_)
    print("Best score: ", model.best_score_)

    return model
    


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    #Making predictions on the test set
    predictions = model.predict(X)

    return predictions
    
