"""
Ridge Regression implemented from scratch using NumPy
"""
import numpy as np

class RidgeRegressionCustom:
    """
    Ridge Regression using Normal Equation with L2 regularization
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    method : str, default='normal_equation'
        'normal_equation' or 'gradient_descent'
    """
    
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, method='normal_equation'):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        if self.method == 'normal_equation':
            # Add bias term
            X_b = np.c_[np.ones((n_samples, 1)), X]
            
            # Ridge Regression: theta = (X^T X + alpha*I)^-1 X^T y
            # Create regularization matrix (excluding bias term)
            I = np.eye(n_features + 1)
            I[0, 0] = 0  # Don't regularize bias term
            
            try:
                theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
            
            self.bias = theta[0]
            self.weights = theta[1:]
            
        elif self.method == 'gradient_descent':
            # Initialize weights and bias
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0
            
            # Gradient descent
            for i in range(self.n_iterations):
                # Predictions
                y_pred = self.predict(X)
                
                # Compute gradients with L2 regularization
                dw = (1/n_samples) * (X.T @ (y_pred - y) + self.alpha * self.weights)
                db = (1/n_samples) * np.sum(y_pred - y)
                
                # Update weights
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Track cost (including regularization term)
                mse = np.mean((y_pred - y)**2)
                l2_penalty = self.alpha * np.sum(self.weights**2)
                cost = mse + l2_penalty
                self.cost_history.append(cost)
                
                # Early stopping if cost doesn't improve
                if i > 10 and abs(self.cost_history[-1] - self.cost_history[-10]) < 1e-8:
                    break
        
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