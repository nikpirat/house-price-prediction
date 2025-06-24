from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Prepares house pricing dataset for modeling by automatically cleaning and transforming its features
# This setup makes machine learning pipeline cleaner, repeatable, and far easier to deploy.
def build_preprocessor(X):
    # Scan dataset X to separate numeric and categorical columns.
    # This lets apply different cleaning strategies depending on column type.
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define the numeric pipeline
    # Fills in missing values in numeric columns using the mean of that column.
    # This keeps model training from crashing due to NaNs.
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean'))
    ])

    # Define the categorical pipeline
    # First, fills missing values in string-based columns with the most frequent category (like a mode).
    # Then, converts those categories into one-hot encoded vectors (like turning "Neighborhood" into binary columns for each area).
    # handle_unknown='ignore' ensures your model doesn’t fail if it encounters a category during inference it didn’t see during training.
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine 2 features into a ColumnTransformer
    # This merges both pipelines into one unified transformer, applying:
    # Numeric logic to numeric columns
    # Categorical logic to categorical columns And leaves the rest untouched.
    preprocessor = ColumnTransformer([
        ('numeric_features', numeric_pipeline, numeric_features),
        ('categorical_features', categorical_pipeline, categorical_features)
    ])

    return preprocessor