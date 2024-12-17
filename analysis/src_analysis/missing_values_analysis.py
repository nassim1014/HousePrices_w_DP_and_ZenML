import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass
    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass

class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df):
        missing_counts = df.isnull().sum()  # Count missing values
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Values': missing_counts.values
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]  # Only show columns with missing values
        
        if missing_df.empty:
            print("No missing values found in the DataFrame.")
        else:
            print("Missing Values Summary:")
            print(missing_df)
        return missing_df
    
    def visualize_missing_values(self,df):
        plt.figure(figsize=(12, 8))  # Set figure size
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title("Missing Values Heatmap")
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, None],
        'C': [1, None, None, 4, 5],
        'D': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    missing_values_analyser = SimpleMissingValuesAnalysis()

    # Identify missing values
    missing_values_analyser.identify_missing_values(df)
    
    # Visualize missing values
    missing_values_analyser.visualize_missing_values(df)