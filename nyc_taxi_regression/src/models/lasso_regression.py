"""
Lasso Regression implemented from scratch using NumPy
"""
import numpy as np

class LassoRegressionCustom:
    """
    Lasso Regression using Coordinate Descent with L1 regularization
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    max_iterations : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    """
    
    def __init__(self, alpha=1.0, max_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def soft_threshold(self, x, threshold):
        """
        Soft thresholding operator for L1 regularization
        
        Parameters:
        -----------
        x : float
        threshold : float
        
        Returns:
        --------
        soft_thresholded_value : float
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0
    
    def fit(self, X, y):
        """
        Train the model using coordinate descent
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = float(np.mean(y))  # Initialize bias to mean of target
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        y_centered = y - self.bias
        
        # Precompute X^T X and X^T y
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        # Coordinate descent
        for iteration in range(self.max_iterations):
            weights_old = self.weights.copy()
            
            # Update each weight coordinate-wise
            for j in range(n_features):
                # Compute residual excluding current feature
                Xj = X_centered[:, j]
                
                # Create mask to exclude current feature
                mask = np.ones(n_features, dtype=bool)
                mask[j] = False
                
                X_not_j = X_centered[:, mask]
                w_not_j = self.weights[mask].copy()
                
                # Compute residual
                residual = y_centered - X_not_j @ w_not_j
                
                # Compute gradient for current feature
                rho = Xj.T @ residual
                
                # Soft thresholding
                threshold = self.alpha * n_samples
                if rho > threshold:
                    self.weights[j] = (rho - threshold) / (Xj.T @ Xj)
                elif rho < -threshold:
                    self.weights[j] = (rho + threshold) / (Xj.T @ Xj)
                else:
                    self.weights[j] = 0.0
            
            # Check convergence
            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break
            
            # Track cost
            y_pred = self.predict(X)
            mse = np.mean((y - y_pred)**2)
            l1_penalty = self.alpha * np.sum(np.abs(self.weights))
            cost = mse + l1_penalty
            self.cost_history.append(cost)
        
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
        return X @ self.weights + self.bias
    
    def get_coefficients(self):
        """Return feature coefficients"""
        return self.weights
    
    def get_bias(self):
        """Return bias term"""
        return self.bias
    
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
    
    def get_alpha(self):
        """Return regularization strength"""
        return self.alpha
    
    def get_feature_importance(self):
        """
        Get feature importance based on absolute coefficient values
        
        Returns:
        --------
        importance : numpy array
        """
        return np.abs(self.weights)