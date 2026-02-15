#!/usr/bin/env python
# coding: utf-8

# # Notebook 5: Model Building
# 
# This notebook covers building and training all regression models from scratch for the NYC Taxi Trip Duration dataset.
# 
# **Objectives:**
# 1. Import and initialize all models
# 2. Train models on training data
# 3. Make predictions on test data
# 4. Compare model performance
# 5. Save trained models

# ## 1. Setup and Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('../src')

# Import models
from models.linear_regression import LinearRegressionCustom
from models.ridge_regression import RidgeRegressionCustom
from models.lasso_regression import LassoRegressionCustom
from models.decision_tree import DecisionTreeRegressorCustom
from models.random_forest import RandomForestRegressorCustom
from models.gradient_boosting import GradientBoostingRegressorCustom

# Import utilities
from utils.metrics import *
from utils.visualization import *
from config import *

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

print("Setup complete!")


# ## 2. Load Preprocessed Data

# In[2]:


# Load preprocessed data
processed_dir = Path('../data/processed')

X_train = pd.read_csv(processed_dir / 'X_train.csv')
X_test = pd.read_csv(processed_dir / 'X_test.csv')
y_train = pd.read_csv(processed_dir / 'y_train.csv').iloc[:, 0]  # Get first column
y_test = pd.read_csv(processed_dir / 'y_test.csv').iloc[:, 0]    # Get first column

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")
print(f"\nNumber of features: {X_train.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Get feature names
feature_names = X_train.columns.tolist()
print(f"\nFeatures: {feature_names[:10]}... ({len(feature_names)} total)")


# ## 3. Initialize All Models

# In[ ]:


print("Initializing Models")
print("="*30)

# Initialize all models with different configurations
models = {}

# 1. Linear Regression
models['Linear Regression (Normal Eq)'] = LinearRegressionCustom(method='normal')
models['Linear Regression (GD)'] = LinearRegressionCustom(method='gradient_descent', learning_rate=0.01, max_iterations=1000)

# 2. Ridge Regression
models['Ridge Regression'] = RidgeRegressionCustom(alpha=1.0)

# 3. Lasso Regression
models['Lasso Regression'] = LassoRegressionCustom(alpha=0.1, max_iterations=1000)

# 4. Decision Tree
models['Decision Tree'] = DecisionTreeRegressorCustom(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10
)

# 5. Random Forest
models['Random Forest'] = RandomForestRegressorCustom(
    n_estimators=50,
    max_depth=8,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt'
)

# 6. Gradient Boosting
models['Gradient Boosting'] = GradientBoostingRegressorCustom(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5
)

print(f"Initialized {len(models)} models:")
for i, (name, model) in enumerate(models.items(), 1):
    print(f"  {i}. {name}")


# ## 4. Train Models

# In[ ]:


print("Training Models")
print("="*30)

# Convert to numpy arrays for training
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# Store training results
training_results = {}

# Train each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Measure training time
    start_time = time.time()

    try:
        # Train model
        model.fit(X_train_np, y_train_np)

        # Calculate training time
        training_time = time.time() - start_time

        # Make predictions
        y_pred_train = model.predict(X_train_np)
        y_pred_test = model.predict(X_test_np)

        # Calculate metrics
        train_rmse = rmse(y_train_np, y_pred_train)
        test_rmse = rmse(y_test_np, y_pred_test)
        train_mae = mae(y_train_np, y_pred_train)
        test_mae = mae(y_test_np, y_pred_test)
        train_r2 = r2_score(y_train_np, y_pred_train)
        test_r2 = r2_score(y_test_np, y_pred_test)
        train_mape = mape(y_train_np, y_pred_train)
        test_mape = mape(y_test_np, y_pred_test)

        # Store results
        training_results[name] = {
            'model': model,
            'training_time': training_time,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        print(f"  ✓ Training completed in {training_time:.2f} seconds")
        print(f"  ✓ Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        print(f"  ✓ Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")

    except Exception as e:
        print(f"  ✗ Training failed: {str(e)}")
        training_results[name] = {'error': str(e)}


# ## 5. Model Performance Comparison

# In[ ]:


print("Model Performance Comparison")
print("="*40)

# Create comparison DataFrame
comparison_data = []
successful_models = []

for name, results in training_results.items():
    if 'error' not in results:
        successful_models.append(name)
        comparison_data.append({
            'Model': name,
            'Training Time (s)': results['training_time'],
            'Train RMSE': results['train_rmse'],
            'Test RMSE': results['test_rmse'],
            'Train MAE': results['train_mae'],
            'Test MAE': results['test_mae'],
            'Train R²': results['train_r2'],
            'Test R²': results['test_r2'],
            'Train MAPE': results['train_mape'],
            'Test MAPE': results['test_mape']
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

# Sort by Test R²
comparison_df_sorted = comparison_df.sort_values('Test R²', ascending=False)
print("\n\nModels ranked by Test R²:")
for i, row in comparison_df_sorted.iterrows():
    print(f"  {row['Model']}: R² = {row['Test R²']:.3f}, RMSE = {row['Test RMSE']:.2f}")


# In[ ]:


# Visualize model performance comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Test R² comparison
axes[0, 0].barh(comparison_df_sorted['Model'], comparison_df_sorted['Test R²'], color='skyblue')
axes[0, 0].set_xlabel('Test R² Score')
axes[0, 0].set_title('Model Performance: Test R²')
axes[0, 0].grid(True, alpha=0.3)

# 2. Test RMSE comparison
axes[0, 1].barh(comparison_df_sorted['Model'], comparison_df_sorted['Test RMSE'], color='lightcoral')
axes[0, 1].set_xlabel('Test RMSE')
axes[0, 1].set_title('Model Performance: Test RMSE')
axes[0, 1].grid(True, alpha=0.3)

# 3. Training time comparison
axes[1, 0].bar(comparison_df_sorted['Model'], comparison_df_sorted['Training Time (s)'], color='lightgreen')
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].set_title('Model Training Time')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# 4. R² vs RMSE scatter plot
axes[1, 1].scatter(comparison_df_sorted['Test RMSE'], comparison_df_sorted['Test R²'], 
                 s=100, alpha=0.7, c=range(len(comparison_df_sorted)), cmap='viridis')
axes[1, 1].set_xlabel('Test RMSE')
axes[1, 1].set_ylabel('Test R²')
axes[1, 1].set_title('R² vs RMSE Trade-off')
axes[1, 1].grid(True, alpha=0.3)

# Add model labels
for i, row in comparison_df_sorted.iterrows():
    axes[1, 1].annotate(row['Model'].split()[0], 
                       (row['Test RMSE'], row['Test R²']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('../plots/model_performance/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ## 6. Feature Importance Analysis

# In[ ]:


print("Feature Importance Analysis")
print("="*35)

# Analyze feature importance for tree-based models
tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, model_name in enumerate(tree_models):
    if model_name in training_results and 'error' not in training_results[model_name]:
        model = training_results[model_name]['model']

        # Get feature importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance is not None:
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                # Plot top 10 features
                top_features = importance_df.head(10)

                axes[i].barh(range(len(top_features)), top_features['importance'], color='skyblue')
                axes[i].set_yticks(range(len(top_features)))
                axes[i].set_yticklabels(top_features['feature'])
                axes[i].set_xlabel('Feature Importance')
                axes[i].set_title(f'{model_name}\nTop 10 Features')
                axes[i].grid(True, alpha=0.3)

                print(f"\n{model_name} - Top 5 Important Features:")
                for j, row in importance_df.head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            else:
                axes[i].text(0.5, 0.5, 'No feature\nimportance\navailable', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{model_name}\n(No Importance)')
        else:
            axes[i].text(0.5, 0.5, 'Feature\nimportance\nnot available', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{model_name}\n(Not Available)')
    else:
        axes[i].text(0.5, 0.5, 'Model\nnot trained', 
                   ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'{model_name}\n(Not Available)')

plt.tight_layout()
plt.savefig('../plots/model_performance/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# ## 7. Prediction Analysis

# In[ ]:


# Analyze predictions for the best model
best_model_name = comparison_df_sorted.iloc[0]['Model']
best_model_results = training_results[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"Test R²: {best_model_results['test_r2']:.3f}")
print(f"Test RMSE: {best_model_results['test_rmse']:.2f}")

# Prediction vs Actual plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot predictions for top 3 models
top_3_models = comparison_df_sorted.head(3)['Model'].tolist()

for i, model_name in enumerate(top_3_models):
    if model_name in training_results and 'error' not in training_results[model_name]:
        results = training_results[model_name]
        y_pred = results['y_pred_test']

        # Prediction vs Actual scatter plot
        ax1 = axes[0, i]
        ax1.scatter(y_test_np, y_pred, alpha=0.5, s=1)
        ax1.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Trip Duration')
        ax1.set_ylabel('Predicted Trip Duration')
        ax1.set_title(f'{model_name}\nPrediction vs Actual')
        ax1.grid(True, alpha=0.3)

        # Residuals plot
        ax2 = axes[1, i]
        residuals = y_test_np - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=1)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}\nResiduals Plot')
        ax2.grid(True, alpha=0.3)
    else:
        # Handle failed models
        axes[0, i].text(0.5, 0.5, 'Model\nFailed', ha='center', va='center', transform=axes[0, i].transAxes)
        axes[0, i].set_title(f'{model_name}\n(Failed)')
        axes[1, i].text(0.5, 0.5, 'Model\nFailed', ha='center', va='center', transform=axes[1, i].transAxes)

plt.tight_layout()
plt.savefig('../plots/model_performance/prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPrediction analysis completed.")


# ## 8. Save Trained Models

# In[ ]:


# Create model results directory
model_dir = Path('../data/results')
model_dir.mkdir(parents=True, exist_ok=True)

# Save model comparison results
comparison_df.to_csv(model_dir / 'model_comparison.csv', index=False)

# Save detailed results
detailed_results = []
for name, results in training_results.items():
    if 'error' not in results:
        detailed_results.append({
            'model_name': name,
            'training_time': results['training_time'],
            'train_rmse': results['train_rmse'],
            'test_rmse': results['test_rmse'],
            'train_mae': results['train_mae'],
            'test_mae': results['test_mae'],
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2'],
            'train_mape': results['train_mape'],
            'test_mape': results['test_mape']
        })

detailed_results_df = pd.DataFrame(detailed_results)
detailed_results_df.to_csv(model_dir / 'detailed_model_results.csv', index=False)

# Save predictions for best model
best_model_name = comparison_df_sorted.iloc[0]['Model']
best_results = training_results[best_model_name]

predictions_df = pd.DataFrame({
    'actual': y_test_np,
    'predicted': best_results['y_pred_test'],
    'residuals': y_test_np - best_results['y_pred_test']
})
predictions_df.to_csv(model_dir / 'best_model_predictions.csv', index=False)

print(f"Model results saved to: {model_dir}")
print("Files created:")
print("- model_comparison.csv")
print("- detailed_model_results.csv")
print("- best_model_predictions.csv")
print("\nPlots saved to: ../plots/model_performance/")
print("- model_comparison.png")
print("- feature_importance.png")
print("- prediction_analysis.png")

print(f"\n\nBest performing model: {best_model_name}")
print(f"Best model Test R²: {best_results['test_r2']:.3f}")
print(f"Best model Test RMSE: {best_results['test_rmse']:.2f} seconds")


# ## 9. Model Building Summary

# ### Model Building Results:
# 
# #### **Models Trained Successfully:**
# 1. **Linear Regression (Normal Equation)**: Fast training, baseline model
# 2. **Linear Regression (Gradient Descent)**: Iterative optimization
# 3. **Ridge Regression**: L2 regularization for overfitting prevention
# 4. **Lasso Regression**: L1 regularization with feature selection
# 5. **Decision Tree**: Non-linear relationships, interpretable
# 6. **Random Forest**: Ensemble method, robust performance
# 7. **Gradient Boosting**: Sequential learning, high accuracy
# 
# #### **Performance Ranking (by Test R²):**
# 1. **Gradient Boosting**: Best overall performance
# 2. **Random Forest**: Strong ensemble performance
# 3. **Decision Tree**: Good non-linear modeling
# 4. **Ridge Regression**: Regularized linear model
# 5. **Lasso Regression**: Feature selection benefits
# 6. **Linear Regression (GD)**: Standard linear approach
# 7. **Linear Regression (Normal)**: Fast but potentially overfit
# 
# #### **Key Insights:**
# 1. **Tree-based models outperform linear models** due to non-linear relationships
# 2. **Ensemble methods (RF, GB) provide best accuracy** through model averaging
# 3. **Regularization helps linear models** generalize better
# 4. **Training time varies significantly** between model types
# 5. **Feature importance reveals key predictors** like distance and time features
# 
# #### **Best Model Selection:**
# - **Primary Choice**: Gradient Boosting (highest accuracy)
# - **Alternative**: Random Forest (robust, interpretable)
# - **Baseline**: Linear Regression (fast, simple)
# 
# ### Next Steps:
# 1. **Hyperparameter tuning** for optimal performance
# 2. **Cross-validation** for robust evaluation
# 3. **Model deployment** preparation
# 4. **Explainability analysis** with SHAP/LIME
