from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Creates a machine learning pipeline that consists of 2 steps
def build_model(preprocessor):
    model = Pipeline([
        ('preprocessor', preprocessor), # data transformation pipeline
        ('regressor', LinearRegression()) # Linear Regression model that learns patterns in the cleaned dataset and makes predictions
    ])
    return model

# Why use a pipeline?

# Cleaner code: everything bundled together
# No manual preprocessing every time you predict
# Prevents data leakage by ensuring preprocessing only happens on training data during fit
# Can be saved and reused later with all steps intact