# German Credit Risk Prediction

This project uses the German Credit dataset to predict credit risk using a Random Forest model. The model is built using scikit-learn, and SHAP (SHapley Additive exPlanations) is used for model interpretation and feature importance analysis.

## Setup Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/Lalanne0/german-credit-dataset-shap.git
cd german-credit-dataset-shap
```

### Step 2: Set up a virtual environment

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the script
```bash
python src/shap_example.py
```

### Step 5: Deactivate the environment

```bash
deactivate
```

## Project Description
### Data
The project uses the German Credit dataset, which includes features such as Age, Sex, Job, Housing, Saving Accounts, Checking account, Credit amount, Duration, and Purpose. The target variable, Risk, is constructed based on credit amount and duration, where higher values indicate higher risk.

### Model Pipeline
The model pipeline includes the following steps:

#### Preprocessing:

Numeric features are imputed and scaled.
Categorical features are imputed and one-hot encoded.

#### Modeling:

A RandomForestClassifier is used to predict the Risk.

#### SHAP Analysis:

SHAP values are computed to explain the model's predictions and to identify the most important features.
### Results
The script generates the following outputs:  
Accuracy: 0.98  
F1-score: 0.98  

SHAP Summary Plot: A plot showing each feature's importance as well as the most important categories among a specific feature.  
Top 10 Features: A list of the top 10 most important features based on SHAP values, with credit duration and amount being the two most important values, as the risk target has been built from them.

## Acknowledgements

This project uses the German Credit Risk dataset available on [Kaggle](https://www.kaggle.com/datasets/uciml/german-credit/).
