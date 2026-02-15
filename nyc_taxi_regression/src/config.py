"""
Configuration file for NYC Taxi Regression project
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'NYC.csv'
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
PLOTS_PATH = PROJECT_ROOT / 'plots'

# Random seed for reproducibility
RANDOM_STATE = 42

# Model hyperparameters (defaults)
LINEAR_REGRESSION_PARAMS = {
    'learning_rate': 0.01,
    'n_iterations': 1000
}

RIDGE_REGRESSION_PARAMS = {
    'alpha': 1.0,
    'learning_rate': 0.01,
    'n_iterations': 1000
}

LASSO_REGRESSION_PARAMS = {
    'alpha': 1.0,
    'max_iterations': 1000,
    'tol': 1e-4
}

DECISION_TREE_PARAMS = {
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'max_features': 'sqrt'
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 20
}

# Feature engineering
NYC_CENTER = (40.758, -73.9855)  # Times Square coordinates
AIRPORTS = {
    'JFK': (40.6413, -73.7781),
    'LaGuardia': (40.7769, -73.8740),
    'Newark': (40.6895, -74.1745)
}

# Preprocessing
TEST_SIZE = 0.2
OUTLIER_THRESHOLD = {
    'trip_duration_min': 60,  # 1 minute
    'trip_duration_max': 10800  # 3 hours
}