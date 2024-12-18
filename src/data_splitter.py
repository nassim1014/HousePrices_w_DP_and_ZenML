from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
class DataSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    @abstractmethod
    def split_data(self, df, target_column):
        pass

# Simple Train-Test Split Strategy
class SimpleTrainTestSplit(DataSplitter):
    """Concrete strategy for simple train-test split"""
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df, target_column):
        """
        Implement simple train-test split
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

# Stratified Train-Test Split Strategy
class StratifiedTrainTestSplit(DataSplitter):
    """Concrete strategy for stratified train-test split"""
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df, target_column):
        """
        Implement stratified train-test split
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

# Time-Series Split Strategy
class TimeSeriesSplitStrategy(DataSplitter):
    """Concrete strategy for time-series split"""
    def __init__(self, train_size_ratio=0.8):
        self.train_size_ratio = train_size_ratio
    
    def split_data(self, df, target_column):
        """
        Implement Time-Series Split
        """
        split_index = int(len(df) * self.train_size_ratio)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        return X_train, X_test, y_train, y_test

# Random Undersampling Split Strategy
class RandomUndersamplingSplit(DataSplitter):
    """Concrete strategy for random undersampling split"""
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df, target_column):
        """
        Implement random undersampling split
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Get class counts
        class_counts = y.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        # Undersample majority class
        minority_data = df[df[target_column] == minority_class]
        majority_data = df[df[target_column] == majority_class].sample(
            n=len(minority_data),
            random_state=self.random_state
        )
        
        # Combine and shuffle
        undersampled_df = pd.concat([minority_data, majority_data]).sample(
            frac=1,
            random_state=self.random_state
        )
        
        X = undersampled_df.drop(columns=[target_column])
        y = undersampled_df[target_column]
        
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

# Context for Data Splitting
class DataSplitterContext:
    """Context for data splitting strategies"""
    def __init__(self, strategy: DataSplitter):
        self.strategy = strategy
    
    def split_data(self, df, target_column):
        return self.strategy.split_data(df, target_column)

# Example Usage
'''
if __name__ == "__main__":
    import pandas as pd
    
    # Example dataframe
    data = {
        'feature1': range(1, 101),
        'feature2': np.random.randn(100),
        'target': [0 if x < 70 else 1 for x in range(100)]
    }
    df = pd.DataFrame(data)
    
    # Simple Train-Test Split
    context = DataSplitterContext(SimpleTrainTestSplit(test_size=0.3))
    X_train, X_test, y_train, y_test = context.split_data(df, target_column='target')
    print("Simple Split: ", len(X_train), len(X_test))

    # Stratified Train-Test Split
    context = DataSplitterContext(StratifiedTrainTestSplit(test_size=0.3))
    X_train, X_test, y_train, y_test = context.split_data(df, target_column='target')
    print("Stratified Split: ", len(X_train), len(X_test))

    # Time-Series Split
    context = DataSplitterContext(TimeSeriesSplitStrategy(train_size_ratio=0.7))
    X_train, X_test, y_train, y_test = context.split_data(df, target_column='target')
    print("Time-Series Split: ", len(X_train), len(X_test))

    # Random Undersampling Split
    context = DataSplitterContext(RandomUndersamplingSplit(test_size=0.3))
    X_train, X_test, y_train, y_test = context.split_data(df, target_column='target')
    print("Random Undersampling Split: ", len(X_train), len(X_test))
'''