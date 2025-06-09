import pytest
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Test case for valid input
    input_data = {
        'PatientId': [1, 2],
        'AppointmentID': [101, 102],
        'Gender': ['F', 'M'],
        'ScheduledDay': ['2023-01-01', '2023-01-02'],
        'AppointmentDay': ['2023-01-05', '2023-01-06'],
        'Age': [30, 45],
        'Neighbourhood': ['A', 'B'],
        'Scholarship': [0, 1],
        'Hypertension': [0, 1],
        'Diabetes': [0, 0],
        'Alcoholism': [0, 1],
        'Handcap': [0, 0],
        'SMS_received': [1, 0],
        'No-show': ['No', 'Yes']
    }
    
    processed_data = preprocess_data(input_data)
    
    assert processed_data is not None
    assert 'Age' in processed_data.columns
    assert processed_data['No-show'].nunique() == 2  # Check unique values in target column

def test_preprocess_data_empty_input():
    # Test case for empty input
    input_data = {}
    
    with pytest.raises(ValueError):
        preprocess_data(input_data)

def test_preprocess_data_invalid_data_types():
    # Test case for invalid data types
    input_data = {
        'PatientId': ['a', 'b'],
        'AppointmentID': [101, 102],
        'Gender': ['F', 'M'],
        'ScheduledDay': ['2023-01-01', '2023-01-02'],
        'AppointmentDay': ['2023-01-05', '2023-01-06'],
        'Age': ['thirty', 'forty-five'],
        'Neighbourhood': ['A', 'B'],
        'Scholarship': [0, 1],
        'Hypertension': [0, 1],
        'Diabetes': [0, 0],
        'Alcoholism': [0, 1],
        'Handcap': [0, 0],
        'SMS_received': [1, 0],
        'No-show': ['No', 'Yes']
    }
    
    with pytest.raises(TypeError):
        preprocess_data(input_data)