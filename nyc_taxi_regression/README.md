# NYC Taxi Trip Duration Regression Analysis

End-to-end regression modeling pipeline with models built from scratch.

## Project Overview

This project implements a comprehensive regression analysis pipeline for predicting NYC Taxi Trip Duration using models built entirely from scratch with NumPy. The project follows a modular structure with detailed Jupyter notebooks for each stage of the machine learning pipeline.

## Project Structure

```
nyc_taxi_regression/
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_eda_and_visualization.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_data_preprocessing.ipynb
│   ├── 05_model_building.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_explainability_and_insights.ipynb
│
├── src/                # Source code
│   ├── models/         # Model implementations (from scratch)
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   ├── ridge_regression.py
│   │   ├── lasso_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   │
│   ├── utils/          # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   │   └── feature_engineering.py
│   │
│   └── config.py
│
├── data/               # Data directory
│   ├── raw/            # Original dataset
│   ├── processed/      # Processed datasets
│   └── results/        # Model results and predictions
│
├── plots/              # Saved visualizations
│   ├── eda/            # Exploratory data analysis plots
│   ├── model_performance/  # Model comparison plots
│   └── explainability/     # Feature importance and explainability plots
│
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Models Implemented (From Scratch)

All models are implemented using only NumPy for mathematical operations:

1. **Linear Regression**
   - Normal Equation method
   - Gradient Descent method
   - Feature coefficient interpretation

2. **Ridge Regression**
   - L2 regularization
   - Normal Equation with regularization
   - Hyperparameter tuning

3. **Lasso Regression**
   - L1 regularization
   - Coordinate Descent algorithm
   - Feature selection capabilities

4. **Decision Tree Regressor**
   - CART algorithm implementation
   - Recursive tree building
   - Feature importance calculation

5. **Random Forest Regressor**
   - Bagging ensemble method
   - Random subspace feature selection
   - Out-of-bag (OOB) score calculation

6. **Gradient Boosting Regressor**
   - Sequential boosting algorithm
   - Pseudo-residuals calculation
   - Learning rate optimization

## Key Features

- ✅ **Complete EDA** with 15+ visualizations
- ✅ **Feature engineering** with distance calculations and temporal features
- ✅ **All models built from scratch** using only NumPy
- ✅ **SHAP and LIME explainability** for tree-based models
- ✅ **Comprehensive model evaluation** and comparison
- ✅ **Business insights and recommendations**
- ✅ **Modular and reusable code**
- ✅ **Detailed documentation** in every notebook

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nyc_taxi_regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the NYC Taxi dataset at:
```
data/raw/nyc_taxi.csv
```

## Usage

1. **Start with Notebook 01** for data loading and exploration
2. **Proceed sequentially** through notebooks 02-07
3. **Each notebook is standalone** and well-documented
4. **All visualizations are saved** to appropriate directories

## Dataset

The project uses the NYC Taxi Trip Duration dataset with the following key information:

- **Target variable**: `trip_duration` (in seconds)
- **Features**: Pickup/dropoff coordinates, datetime, passenger count, vendor info
- **Goal**: Predict trip duration based on various features

## Technical Implementation

### Core Technologies
- **NumPy**: Mathematical operations and linear algebra
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive analysis and documentation

### Key Algorithms Implemented
- **Linear Algebra**: Matrix operations, eigenvalues, SVD
- **Optimization**: Gradient descent, coordinate descent
- **Statistics**: Correlation analysis, hypothesis testing
- **Machine Learning**: Ensemble methods, regularization

## Results and Insights

The project provides:
- **Model performance comparison** across all algorithms
- **Feature importance analysis** using multiple methods
- **SHAP and LIME explanations** for model predictions
- **Business recommendations** for taxi operations
- **Deployment considerations** and monitoring strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or feedback, please open an issue or contact the project maintainer.

---

**This project serves as a comprehensive reference for regression modeling, feature engineering, and ML explainability using models built from scratch.**