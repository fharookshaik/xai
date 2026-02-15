"""
Evaluation metrics implemented from scratch
"""
import numpy as np

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    
    Returns:
    --------
    float
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    
    Returns:
    --------
    float
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    R-squared (coefficient of determination)
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    
    Returns:
    --------
    float
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    
    Returns:
    --------
    float
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Comprehensive model evaluation
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    model_name : str
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {
        'Model': model_name,
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }
    return metrics