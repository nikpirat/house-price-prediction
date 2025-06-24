import joblib
import pandas as pd
from config import MODEL_PATH

# 📈 Predict house prices for a batch of inputs (used with CSV)
def predict_house_batch(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(df)
    return predictions