"""
Preprocessing utilities implemented from scratch
"""
import numpy as np

class StandardScalerCustom:
    """
    Standardize features by removing mean and scaling to unit variance
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """
        Compute mean and std from training data
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        """
        Apply standardization
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        
        Returns:
        --------
        X_scaled : numpy array, shape (n_samples, n_features)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Parameters:
        -----------
        X : numpy array
        
        Returns:
        --------
        X_scaled : numpy array
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Reverse standardization
        """
        return X * self.std_ + self.mean_


def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    test_size : float, default=0.2
    random_state : int, default=None
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Random permutation of indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test