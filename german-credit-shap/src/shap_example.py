import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
from sklearn.metrics import f1_score

# Load the German Credit dataset in the data forlder
data = pd.read_csv('data/german_credit_data.csv', index_col=0)

# Construct the target variable
data['Risk'] = ((data['Credit amount'] > data['Credit amount'].median()) & 
                (data['Duration'] > data['Duration'].median())).astype(int)

# Prepare the features and target
X = data.drop(['Risk'], axis=1)
y = data['Risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = ['Age', 'Credit amount', 'Duration']
categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

model.fit(X_train, y_train)

# Get feature names after preprocessing
if hasattr(model['preprocessor'].named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
    get_feature_names = lambda x: model['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(x)
else:
    get_feature_names = lambda x: model['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names(x)

feature_names = numeric_features + list(get_feature_names(categorical_features))

# Print model accuracy
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

# Print F1 score
print(f"F1 score: {f1_score(y_test, model.predict(X_test)):.2f}")

# SHAP analysis
explainer = shap.TreeExplainer(model['classifier'])

# Transform X_test using the preprocessor
X_test_transformed = model['preprocessor'].transform(X_test)

# Compute SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Select SHAP values for the positive class (class 1)
shap_values_class1 = shap_values[:, :, 1]

# Create the SHAP summary plot
shap.summary_plot(shap_values_class1, X_test_transformed, feature_names=feature_names)


# Print top 10 most important features based on SHAP values for class 1
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': np.abs(shap_values_class1).mean(0)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features (based on SHAP values):")
print(feature_importance.head(10))