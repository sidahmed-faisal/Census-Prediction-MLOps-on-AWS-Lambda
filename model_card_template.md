# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is using XGBoost Classifer uses Randomized Cross-Validation Search over 5 folds with set of best set of hyperparameters:  {'learning_rate': 0.10429150466596819, 'max_depth': 9, 'n_estimators': 125, 'subsample': 0.9286585974385411}
with Best score:  0.8675864799692512
## Intended Use
This model predicts whether a person earns over 50k or not based on the census data.
## Training Data
More details about the training data: https://archive.ics.uci.edu/ml/datasets/census+income

Extraction was done by Barry Becker from the 1994 Census database.

Prediction task is to determine whether a person makes over 50K a year.

Features:

* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct,  Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
For both training and evaluation, categorical features of the data are encoded using OneHotEncoder and the target is transformed using LabelBinarizer
## Evaluation Data
The original dataset is first preprocessed and then split into training and evaluation data with evaluation data size of 20% and 80% of training
## Metrics
Performances of the model:

precision: 0.773038842345773
recall: 0.6757656458055925
fbeta: 0.7211367673179396

## Ethical Considerations
This model has been trained using census data and is impartial towards any specific demographic group.
## Caveats and Recommendations
Extraction was performed using the 1994 Census database. The dataset represents an outdated sample and should not be relied upon as a statistically accurate representation of the population. It is advisable to utilize this dataset for training purposes in machine learning classification or similar tasks.
