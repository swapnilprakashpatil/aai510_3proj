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

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset."""
    print("[preprocessing] Starting preprocessing...")
    print("Initial shape of the dataset:", df.shape)
    print("Initial columns in the dataset:", df.columns)

    # Drop unnecessary columns
    print("Dropping unnecessary columns...")
    print("Before dropping:", df.columns)
    print(df.columns)
    columns_to_drop = ['AppointmentID', 'PatientId']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print("After dropping:", df.columns)
    print(df.columns)

    # Convert date columns to datetime
    print("Converting date columns to datetime...")
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
    df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

    # Handle missing values
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    print("[preprocessing] Missing values handled.")
    
    # # Convert categorical variables using one-hot encoding
    # print("Encoding categorical variables...")
    # categorical_cols = ['Gender', 'Neighbourhood', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
    # encoder = OneHotEncoder(sparse_output=False) 
    # encoded_cats = encoder.fit_transform(df[categorical_cols])
    # encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    # print("[preprocessing] Categorical variables encoded.")
    
    # # Drop original categorical columns
    # print("Dropping original categorical columns...")
    # df = df.drop(categorical_cols, axis=1)
    
    # #Concatenate the encoded categorical columns with the original dataframe
    # df = pd.concat([df, encoded_df], axis=1)
    # print("[preprocessing] Encoded columns concatenated.")

    # Add emotional state columns from PatientSentiment
    for emotion in config.EMOTION_STATES:
        df[emotion] = df['PatientSentiment'].str.contains(emotion, case=False, na=False).astype(int)
    print("[preprocessing] Emotional state columns added:", config.EMOTION_STATES)
    
    # Handle any remaining missing values
    print("Final shape of the dataset:", df.shape)
    print("Final columns in the dataset:", df.columns)    
    print("[preprocessing] Preprocessing complete.")
    return df

def split_data(df, target_column):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)