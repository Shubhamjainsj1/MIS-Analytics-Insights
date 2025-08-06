"""
Configuration file for Customer Sentiment Analysis Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    'server': 'localhost',
    'database': 'customer_complaints',
    'username': 'your_username',
    'password': 'your_password',
    'driver': 'ODBC Driver 17 for SQL Server'
}

# Spark configuration
SPARK_CONFIG = {
    'app_name': 'CustomerSentimentAnalysis',
    'master': 'local[*]',
    'memory': '4g',
    'cores': '2'
}

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'sentiment_threshold': 0.1
}

# File paths
SAMPLE_DATA_PATH = RAW_DATA_DIR / "customer_complaints.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_complaints.csv"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"
ESCALATION_MODEL_PATH = MODELS_DIR / "escalation_model.pkl"
