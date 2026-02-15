"""
Decision Tree Regressor implemented from scratch using NumPy
"""
import numpy as np
from collections import Counter

class DecisionTreeRegressorCustom:
    """
    Decision Tree Regressor using CART algorithm
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=10
        Minimum number of samples required to be at a leaf node
    """
    
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_importance_ = None
    
    def _calculate_mse(self, y):
        """Calculate Mean Squared Error"""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _best_split(self, X, y):
        """Find the best split for a node"""
        best_mse = float('inf')
        best_split = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Try all possible split points
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                # Use midpoint as split point
                split_value = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split the data
                left_mask = feature_values <= split_value
                right_mask = ~left_mask
                
                # Check minimum samples constraints
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                # Calculate weighted MSE
                left_mse = self._calculate_mse(y[left_mask])
                right_mse = self._calculate_mse(y[right_mask])
                weighted_mse = left_mse + right_mse
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_split = {
                        'feature_idx': feature_idx,
                        'split_value': split_value,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                        'left_mean': np.mean(y[left_mask]),
                        'right_mean': np.mean(y[right_mask])
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            
            return {'value': np.mean(y), 'is_leaf': True}
        
        # Find the best split
        best_split = self._best_split(X, y)
        
        if best_split is None:
            return {'value': np.mean(y), 'is_leaf': True}
        
        # Create node
        node = {
            'feature_idx': best_split['feature_idx'],
            'split_value': best_split['split_value'],
            'is_leaf': False,
            'left': None,
            'right': None,
            'left_mean': best_split['left_mean'],
            'right_mean': best_split['right_mean']
        }
        
        # Recursively build left and right subtrees
        node['left'] = self._build_tree(
            X[best_split['left_mask']], 
            y[best_split['left_mask']], 
            depth + 1
        )
        
        node['right'] = self._build_tree(
            X[best_split['right_mask']], 
            y[best_split['right_mask']], 
            depth + 1
        )
        
        return node
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
        y : numpy array, shape (n_samples,)
        """
        self.tree = self._build_tree(X, y)
        self._calculate_feature_importance(X, y)
        return self
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node['is_leaf']:
            return node['value']
        
        feature_idx = node['feature_idx']
        split_value = node['split_value']
        
        if x[feature_idx] <= split_value:
            left_child = node['left']
            if left_child is not None:
                return self._predict_single(x, left_child)
            else:
                return node['left_mean']
        else:
            right_child = node['right']
            if right_child is not None:
                return self._predict_single(x, right_child)
            else:
                return node['right_mean']
    
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
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance"""
        n_features = X.shape[1]
        self.feature_importance_ = np.zeros(n_features)
        
        def traverse_tree(node, X_node, y_node):
            if node['is_leaf']:
                return
            
            # Calculate importance for this split
            feature_idx = node['feature_idx']
            left_mask = X_node[:, feature_idx] <= node['split_value']
            right_mask = ~left_mask
            
            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                # Weight by reduction in MSE
                parent_mse = self._calculate_mse(y_node)
                left_mse = self._calculate_mse(y_node[left_mask])
                right_mse = self._calculate_mse(y_node[right_mask])
                
                importance = parent_mse - (left_mse + right_mse)
                self.feature_importance_[feature_idx] += importance
            
            # Traverse children
            traverse_tree(node['left'], X_node[left_mask], y_node[left_mask])
            traverse_tree(node['right'], X_node[right_mask], y_node[right_mask])
        
        traverse_tree(self.tree, X, y)
        
        # Normalize importance
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ /= np.sum(self.feature_importance_)
    
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