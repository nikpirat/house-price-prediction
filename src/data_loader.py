import pandas as pd
from src.config import DATA_PATH

def load_data(path=DATA_PATH):
    return pd.read_csv(path)