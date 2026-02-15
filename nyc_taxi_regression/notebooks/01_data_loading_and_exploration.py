#!/usr/bin/env python
# coding: utf-8

# # Notebook 1: Data Loading and Exploration
# 
# This notebook covers the initial data loading and exploration phase of our NYC Taxi Trip Duration regression analysis.
# 
# **Objectives:**
# 1. Load the dataset
# 2. Perform initial data quality checks
# 3. Understand the data structure and basic statistics
# 4. Identify potential issues and data types
# 5. Save initial data insights

# ## 1. Setup and Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('../src')

# Import configuration
from config import DATA_PATH, RANDOM_STATE

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Setup complete!")


# ## 2. Load Dataset

# In[2]:


# Check if data file exists
print(f"Looking for data at: {DATA_PATH}")
print(f"File exists: {DATA_PATH.exists()}")

# If the file doesn't exist at the configured path, try to find it
if not DATA_PATH.exists():
    # Look for the file in the current directory or data/raw
    possible_paths = [
        Path("../data/raw/nyc_taxi.csv"),
        Path("./data/raw/nyc_taxi.csv"),
        Path("nyc_taxi.csv"),
        Path("../NYC.csv")
    ]

    for path in possible_paths:
        if path.exists():
            DATA_PATH = path
            print(f"Found data at: {DATA_PATH}")
            break
    else:
        print("Data file not found. Please ensure the NYC Taxi dataset is available.")
        # Create sample data for demonstration
        print("Creating sample data for demonstration purposes...")


# In[3]:


# Load the dataset
try:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Successfully loaded data from {DATA_PATH}")
    else:
        # Create sample data for demonstration
        print("Creating sample NYC Taxi data for demonstration...")

        # Generate sample data
        n_samples = 10000
        np.random.seed(RANDOM_STATE)

        # Generate sample features
        pickup_datetime = pd.date_range('2023-01-01', periods=n_samples, freq='10min')
        dropoff_datetime = pickup_datetime + pd.to_timedelta(np.random.exponential(20, n_samples), unit='m')

        df = pd.DataFrame({
            'id': [f'id_{i}' for i in range(n_samples)],
            'vendor_id': np.random.choice([1, 2], n_samples),
            'pickup_datetime': pickup_datetime,
            'dropoff_datetime': dropoff_datetime,
            'passenger_count': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.6, 0.25, 0.1, 0.03, 0.015, 0.005]),
            'pickup_longitude': np.random.normal(-73.98, 0.05, n_samples),
            'pickup_latitude': np.random.normal(40.75, 0.05, n_samples),
            'dropoff_longitude': np.random.normal(-73.98, 0.05, n_samples),
            'dropoff_latitude': np.random.normal(40.75, 0.05, n_samples),
            'store_and_fwd_flag': np.random.choice(['Y', 'N'], n_samples, p=[0.05, 0.95])
        })

        # Calculate trip duration
        df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

        print(f"Created sample dataset with {n_samples} rows")

    print(f"Dataset shape: {df.shape}")

except Exception as e:
    print(f"Error loading data: {e}")


# ## 3. Display First and Last Rows

# In[4]:


print("First 5 rows:")
print(df.head())
print("\n" + "="*50 + "\n")
print("Last 5 rows:")
print(df.tail())


# ## 4. Dataset Information and Data Types

# In[5]:


print("Dataset Information:")
print("="*50)
df.info()

print("\n\nData Types:")
print("="*50)
print(df.dtypes)

print("\n\nMemory Usage:")
print("="*50)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# ## 5. Column Explanations

# ### Column Descriptions:
# 
# 1. **id**: Unique identifier for each trip
# 2. **vendor_id**: A code indicating the provider associated with the trip record
# 3. **pickup_datetime**: Date and time when the meter was engaged
# 4. **dropoff_datetime**: Date and time when the meter was disengaged
# 5. **passenger_count**: The number of passengers in the vehicle (driver entered value)
# 6. **pickup_longitude**: Longitude where the meter was engaged
# 7. **pickup_latitude**: Latitude where the meter was engaged
# 8. **dropoff_longitude**: Longitude where the meter was disengaged
# 9. **dropoff_latitude**: Latitude where the meter was disengaged
# 10. **store_and_fwd_flag**: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# 11. **trip_duration**: Duration of the trip in seconds (TARGET VARIABLE)

# ## 6. Initial Data Quality Check

# In[6]:


print("Missing Values Analysis:")
print("="*50)
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percent
})
print(missing_df)

print("\n\nDuplicate Rows:")
print("="*50)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print(f"Percentage of duplicates: {(duplicates / len(df)) * 100:.2f}%")


# ## 7. Basic Statistics

# In[7]:


print("Numerical Columns Summary Statistics:")
print("="*50)
numerical_stats = df.describe()
print(numerical_stats)

print("\n\nCategorical Columns Value Counts:")
print("="*50)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))


# ## 8. Data Type Distribution

# In[8]:


print("Data Type Distribution:")
print("="*50)
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

# Visualize data type distribution
plt.figure(figsize=(10, 6))
dtype_counts.plot(kind='bar')
plt.title('Data Type Distribution')
plt.xlabel('Data Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../plots/eda/data_type_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ## 9. Save Initial Data Insights

# In[9]:


# Create processed data directory if it doesn't exist
processed_dir = Path('../data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

# Save dataset info
dataset_info = {
    'shape': df.shape,
    'columns': list(df.columns),
    'dtypes': df.dtypes.to_dict(),
    'missing_values': missing_df.to_dict(),
    'duplicates': duplicates,
    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
}

# Save as CSV for easy reading
df.to_csv(processed_dir / 'initial_dataset.csv', index=False)

# Save summary statistics
numerical_stats.to_csv(processed_dir / 'summary_statistics.csv')

print(f"Dataset info saved to: {processed_dir}")
print("Files created:")
print("- initial_dataset.csv")
print("- summary_statistics.csv")
print("- data_type_distribution.png")


# ## 10. Initial Findings Summary

# ### Key Initial Findings:
# 
# 1. **Dataset Size**: The dataset contains approximately {df.shape[0]:,} rows and {df.shape[1]} columns
# 2. **Data Types**: Mix of numerical and categorical data types
# 3. **Missing Values**: {missing_df['Missing Count'].sum()} total missing values across all columns
# 4. **Duplicates**: {duplicates} duplicate rows found ({(duplicates/len(df))*100:.2f}% of dataset)
# 5. **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
# 
# ### Data Quality Issues Identified:
# - [List specific issues found in the data]
# - [Missing values in specific columns]
# - [Data type inconsistencies]
# - [Potential outliers or invalid values]
# 
# ### Next Steps:
# 1. Proceed to EDA for deeper insights
# 2. Handle missing values and duplicates
# 3. Convert data types as needed
# 4. Feature engineering and preprocessing
