import os

# Paths
DATA_DIR = "."
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_DIR = "outputs"
FOLDS_PATH = os.path.join(OUTPUT_DIR, "train_folds.csv")

# Global Configuration
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = "Will_Buy_EV"
ID_COL = "id"

# Raw Feature Definitions
NUMERICAL_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

BINARY_COLS = [
    "Home_Charging_Possible",
    "Subsidy_Available",
]
