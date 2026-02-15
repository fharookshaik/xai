"""
Visualization utilities for regression analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(feature_names, importances, title='Feature Importance', 
                           figsize=(10, 6), save_path=None):
    """
    Plot feature importance as horizontal bar chart
    
    Parameters:
    -----------
    feature_names : list
    importances : array-like
    title : str
    figsize : tuple
    save_path : str, optional
    """
    # Sort by importance
    indices = np.argsort(importances)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, model_name='Model', 
                               figsize=(8, 8), save_path=None):
    """
    Scatter plot of predictions vs actual values
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Trip Duration (seconds)')
    plt.ylabel('Predicted Trip Duration (seconds)')
    plt.title(f'{model_name}: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, model_name='Model', figsize=(12, 5), save_path=None):
    """
    Plot residuals analysis
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name}: Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name}: Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(data, title='Feature Correlation Matrix', figsize=(12, 10), save_path=None):
    """
    Plot correlation matrix heatmap
    
    Parameters:
    -----------
    data : pandas DataFrame
    title : str
    figsize : tuple
    save_path : str, optional
    """
    plt.figure(figsize=figsize)
    correlation_matrix = data.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_time_series_analysis(data, date_column, value_column, title='Time Series Analysis', 
                             figsize=(14, 6), save_path=None):
    """
    Plot time series analysis
    
    Parameters:
    -----------
    data : pandas DataFrame
    date_column : str
    value_column : str
    title : str
    figsize : tuple
    save_path : str, optional
    """
    plt.figure(figsize=figsize)
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort by date
    data = data.sort_values(date_column)
    
    plt.plot(data[date_column], data[value_column], alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_distribution_analysis(data, column, title='Distribution Analysis', 
                              figsize=(12, 5), save_path=None):
    """
    Plot distribution analysis with histogram and box plot
    
    Parameters:
    -----------
    data : pandas DataFrame
    column : str
    title : str
    figsize : tuple
    save_path : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    axes[0].hist(data[column], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(data[column])
    axes[1].set_ylabel(column)
    axes[1].set_title(f'{title} - Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()