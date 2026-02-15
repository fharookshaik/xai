"""
Random Forest Regressor implemented from scratch using NumPy
"""
import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeRegressorCustom

class RandomForestRegressorCustom:
    """
    Random Forest Regressor using bagging and random subspace method
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=10
        Maximum depth of the trees
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=10
        Minimum number of samples required to be at a leaf node
    max_features : str or int, default='sqrt'
        Number of features to consider for each split
    bootstrap : bool, default=True
        Whether to use bootstrap sampling
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=20, 
                 min_samples_leaf=10, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importance_ = None
        self.oob_score_ = None
    
    def _get_max_features(self, n_features):
        """Get number of features to consider for each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y, random_state):
        """Create bootstrap sample"""
        n_samples = len(y)
        if self.random_state is not None:
            np.random.seed(self.random_state + random_state)
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _random_subspace(self, X, y, max_features, random_state):
        """Select random subset of features"""
        if self.random_state is not None:
            np.random.seed(self.random_state + random_state)
        
        n_features = X.shape[1]
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        return X[:, feature_indices], feature_indices
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        # Initialize feature importance
        self.feature_importance_ = np.zeros(n_features)
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_bootstrap, y_bootstrap, bootstrap_indices = self._bootstrap_sample(X, y, i)
            else:
                X_bootstrap, y_bootstrap = X, y
                bootstrap_indices = np.arange(n_samples)
            
            # Random subspace
            X_subspace, feature_indices = self._random_subspace(
                X_bootstrap, y_bootstrap, max_features, i
            )
            
            # Train decision tree
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            tree.fit(X_subspace, y_bootstrap)
            self.trees.append((tree, feature_indices))
            
            # Calculate feature importance
            tree_importance = tree.get_feature_importance()
            if tree_importance is not None:
                for j, feature_idx in enumerate(feature_indices):
                    self.feature_importance_[feature_idx] += tree_importance[j]
            
            # Calculate OOB predictions
            if self.bootstrap:
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[bootstrap_indices] = False
                oob_indices = np.where(oob_mask)[0]
                
                if len(oob_indices) > 0:
                    X_oob = X[oob_indices][:, feature_indices]
                    oob_pred = tree.predict(X_oob)
                    
                    oob_predictions[oob_indices] += oob_pred
                    oob_counts[oob_indices] += 1
        
        # Normalize feature importance
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ /= np.sum(self.feature_importance_)
        
        # Calculate OOB score
        if self.bootstrap:
            valid_oob = oob_counts > 0
            if np.sum(valid_oob) > 0:
                oob_predictions[valid_oob] /= oob_counts[valid_oob]
                self.oob_score_ = 1 - np.sum((y[valid_oob] - oob_predictions[valid_oob])**2) / np.sum((y[valid_oob] - np.mean(y[valid_oob]))**2)
        
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
        predictions = np.zeros(len(X))
        
        for tree, feature_indices in self.trees:
            X_subspace = X[:, feature_indices]
            tree_predictions = tree.predict(X_subspace)
            predictions += tree_predictions
        
        return predictions / len(self.trees)
    
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