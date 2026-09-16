import pandas as pd
import numpy as np
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS

SELECTED_ENGINEERED_FEATURES = [
    "Subsidy_x_Income",
    "Income_per_Car",
]

ALL_MODEL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS + SELECTED_ENGINEERED_FEATURES

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the validated features that strictly improved 5-fold OOF AUC:
    1. Subsidy_x_Income: captures government financial incentive scaled by income level.
    2. Income_per_Car: household capital / purchasing power per vehicle.
    """
    df = df.copy()
    subsidy_bin = (df["Subsidy_Available"] == "Yes").astype(float)
    df["Subsidy_x_Income"] = subsidy_bin * df["Annual_Income_USD"]
    df["Income_per_Car"] = df["Annual_Income_USD"] / df["Number_of_Cars_Owned"].clip(lower=1)
    return df
