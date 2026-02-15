"""
Gradient Boosting Regressor implemented from scratch using NumPy
"""
import numpy as np
from .decision_tree import DecisionTreeRegressorCustom

class GradientBoostingRegressorCustom:
    """
    Gradient Boosting Regressor using sequential boosting
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages to perform
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree
    max_depth : int, default=5
        Maximum depth of the individual regression estimators
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=10
        Minimum number of samples required to be at a leaf node
    loss : str, default='squared_error'
        Loss function to be optimized
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, 
                 min_samples_split=20, min_samples_leaf=10, 
                 loss='squared_error', random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.random_state = random_state
        self.trees = []
        self.feature_importance_ = None
        self.initial_prediction = None
    
    def _calculate_gradient(self, y, y_pred):
        """Calculate gradient for squared error loss"""
        return 2 * (y_pred - y)
    
    def _calculate_hessian(self, y, y_pred):
        """Calculate hessian for squared error loss"""
        return 2 * np.ones_like(y)
    
    def _calculate_pseudo_residuals(self, y, y_pred):
        """Calculate pseudo residuals"""
        return -self._calculate_gradient(y, y_pred)
    
    def _calculate_gamma(self, y, y_pred, residuals, leaf_indices):
        """Calculate optimal leaf values"""
        if self.loss == 'squared_error':
            # For squared error, gamma is simply the mean of residuals in the leaf
            return np.mean(residuals[leaf_indices])
        else:
            # Newton-Raphson update for other losses
            numerator = np.sum(residuals[leaf_indices])
            denominator = np.sum(self._calculate_hessian(y[leaf_indices], y_pred[leaf_indices]))
            return numerator / denominator if denominator != 0 else 0
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize predictions with target mean
        self.initial_prediction = np.mean(y)
        y_pred = np.full(n_samples, self.initial_prediction)
        
        # Initialize feature importance
        self.feature_importance_ = np.zeros(n_features)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Calculate pseudo residuals
            residuals = self._calculate_pseudo_residuals(y, y_pred)
            
            # Fit tree to residuals
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Make predictions with this tree
            tree_predictions = tree.predict(X)
            
            # Update predictions
            y_pred += self.learning_rate * tree_predictions
            
            # Calculate feature importance
            tree_importance = tree.get_feature_importance()
            if tree_importance is not None:
                self.feature_importance_ += tree_importance
        
        # Normalize feature importance
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ /= np.sum(self.feature_importance_)
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        
        Returns:
        --------
        predictions : numpy array, shape (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)
        
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
        
        return predictions
    
    def get_feature_importance(self):
        """Return feature importance"""
        return self.feature_importance_
    
    def score(self, X, y):
        """
        Calculate R-squared score
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        
        Returns:
        --------
        r2_score : float
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)
    
    def staged_predict(self, X):
        """
        Generate predictions at each stage of boosting
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        
        Returns:
        --------
        predictions : generator of numpy arrays
        """
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)
        
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
            yield predictions.copy()