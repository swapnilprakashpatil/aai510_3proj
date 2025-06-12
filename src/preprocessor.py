import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def load_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[preprocessing] Starting preprocessing...")
        print("Initial shape of the dataset:", df.shape)
        print("Initial columns in the dataset:", df.columns)

        df = self._drop_unnecessary_columns(df)
        df = self._convert_columns(df)
        df = self._handle_missing_values(df)
        df = self._add_emotional_state_columns(df)

        print("Final shape of the dataset:", df.shape)
        print("Final columns in the dataset:", df.columns)
        print("[preprocessing] Preprocessing complete.")
        return df

    def split_data(self, df: pd.DataFrame, target_column: str):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Dropping unnecessary columns...")
        columns_to_drop = ['AppointmentID', 'PatientId']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print("Remaining columns:", df.columns)
        return df

    def _convert_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Converting date columns to datetime...")
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
        df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        df['Age'] = imputer.fit_transform(df[['Age']])
        return df

    def _add_emotional_state_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding emotional state columns...")
        for emotion in self.config.EMOTION_STATES:
            df[emotion] = df['PatientSentiment'].str.contains(emotion, case=False, na=False).astype(int)
        print("Emotional state columns added:", self.config.EMOTION_STATES)
        return df