from src.data_loader import load_data
from src.preprocess import build_preprocessor
from src.model import build_model
from sklearn.model_selection import train_test_split
import joblib
from src.config import MODEL_PATH

# train_model function is the engine that builds, trains, and saves linear regression model for predicting house prices
def train_model():

    # Load dataset
    df = load_data()

    # Split into features and target
    X = df.drop("SalePrice", axis=1) # Contains all except SalePrice
    y = df["SalePrice"] # Contains SalePrice

    # Preprocess features
    preprocessor = build_preprocessor(X) # builds a transformer that Fills in missing values & One-hot encodes categorical variables

    # Build the model pipeline
    model = build_model(preprocessor)

    # Splitting the data:
    # 80% → training
    # 20% → test (held out, not seen during training)
    # This is important for evaluating generalization later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trains the model using just the training data.
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

    return X_test, y_test # return test set for evaluation