import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# load/clean
def load_and_clean_dummy_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dummy_mimic_data.csv')
    data = pd.read_csv(data_path)
    data['admission_type'] = data['admission_type'].fillna('UNKNOWN')
    data['admission_location'] = data['admission_location'].fillna('UNKNOWN')
    data['icd_code'] = data['icd_code'].fillna('UNKNOWN')
    return data

# categorical into vectors
def encode_and_prepare_features(data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Updated argument
    categorical_columns = ['admission_type', 'admission_location', 'gender']
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
    numerical_columns = ['anchor_age']
    numerical_data = data[numerical_columns].reset_index(drop=True)
    prepared_data = pd.concat([encoded_df, numerical_data], axis=1)

    return prepared_data, data['icd_code']

# train, validation, and test sets
def split_data(features, targets):
    targets = targets.astype('category').cat.codes  
    X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# save data as tensors
def save_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    tensor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    torch.save((torch.tensor(X_train.values).float(), torch.tensor(y_train.values).long()), os.path.join(tensor_path, 'train_tensor.pt'))
    torch.save((torch.tensor(X_val.values).float(), torch.tensor(y_val.values).long()), os.path.join(tensor_path, 'val_tensor.pt'))
    torch.save((torch.tensor(X_test.values).float(), torch.tensor(y_test.values).long()), os.path.join(tensor_path, 'test_tensor.pt'))

if __name__ == "__main__":
    dummy_data = load_and_clean_dummy_data()
    features, targets = encode_and_prepare_features(dummy_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets)
    save_tensors(X_train, X_val, X_test, y_train, y_val, y_test)

