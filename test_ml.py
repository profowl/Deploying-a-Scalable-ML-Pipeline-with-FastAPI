import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

import pandas as pd
import os


# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Sample fixture to mock input data
@pytest.fixture
def sample_data():
    """
    Load a small subset of census data for testing.
    """
    project_path = os.getwd()
    print(project_path)
    data_path = os.path.join(
        project_path,
        "data",
        "census.csv"
    )
    data = pd.read_csv(data_path)
    return data.sample(n=100, random_state=42)  # small for testing


# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_classifier(sample_data):
    """
    Test if train_model returns a RandomForestClassifier.
    """
    X, y, _, _ = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_returns_valid_values(sample_data):
    """
    Test if compute_model_metrics returns float values in expected range.
    """
    X, y, _, _ = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0


# TODO: implement the third test. Change the function name and input as needed
def test_process_data_shapes(sample_data):
    """
    Test if process_data returns correct shapes and types.
    """
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 2  # should be 2D matrix
