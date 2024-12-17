from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature : str):
        pass

class NumericalUnivariateAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df, feature):
        """
        Perform numerical univariate analysis for a given feature in a DataFrame.
        
        Parameters:
            feature (str): The name of the numerical column to analyze.
            df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
            summary_stats (pd.Series): A series containing descriptive statistics for the feature.
        """
        # Check if the feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Error: '{feature}' is not in the DataFrame.")
            return
        
        # Check if the feature is numerical
        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Error: '{feature}' is not a numerical column.")
            return
        
        # 1. Descriptive Statistics
        summary_stats = df[feature].describe()
        print(f"### Descriptive Statistics for '{feature}' ###\n{summary_stats}\n")
        
        # 2. Check for Missing Values
        missing_values = df[feature].isnull().sum()
        print(f"Missing values in '{feature}': {missing_values}\n")
        
        # 3. Identify Outliers using IQR (Interquartile Range)
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        
        print(f"Number of outliers in '{feature}': {outliers.shape[0]}\n")
        
        # 4. Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Univariate Analysis for '{feature}'", fontsize=14)
        
        # Histogram
        sns.histplot(df[feature], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("Histogram")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frequency")
        
        # Boxplot
        sns.boxplot(y=df[feature], ax=axes[1], color='lightgreen')
        axes[1].set_title("Boxplot")
        axes[1].set_ylabel(feature)
        
        plt.tight_layout()
        plt.show()
        
        return summary_stats


class CategoricalUnivariateAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df, feature):
        """
        Perform categorical univariate analysis for a given feature in a DataFrame.
        
        Parameters:
            feature (str): The name of the categorical column to analyze.
            df (pd.DataFrame): The DataFrame containing the data.
        
        Returns:
            value_counts (pd.Series): A series showing counts of unique values in the feature.
        """
        # Check if the feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Error: '{feature}' is not in the DataFrame.")
            return
        
        # Check if the feature is categorical
        if not df[feature].dtype == 'object' and not pd.api.types.is_categorical_dtype(df[feature]):
            print(f"Error: '{feature}' is not a categorical column.")
            return
        
        # 1. Value Counts
        value_counts = df[feature].value_counts()
        print(f"### Value Counts for '{feature}' ###\n{value_counts}\n")
        
        # 2. Check for Missing Values
        missing_values = df[feature].isnull().sum()
        print(f"Missing values in '{feature}': {missing_values}\n")
        
        # 3. Visualization - Bar Plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=feature, palette='viridis', order=value_counts.index)
        plt.title(f"Distribution of '{feature}'")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.show()
        
        return value_counts
    
class UnivariateAnalyzer:
    def __init__(self, strategy : UnivariateAnalysisStrategy):
        self._strategy = strategy
    def set_strategy(self, strategy : UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df : pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)



if __name__ == "__main__":
    # Sample DataFrame
    data = {
        'price': [100, 200, 300, 400, 500, 600, 700, 10000],
        'area': [50, 60, 70, 80, 90, 100, 110, 120]
    }
    df = pd.DataFrame(data)

    #analyzer = NumericalUnivariateAnalysisStrategy()
    # Perform univariate analysis on 'price'
    #analyzer.analyze( df, 'price')

# Sample DataFrame
    data = {
        'furnishingstatus': ['furnished', 'semi-furnished', 'unfurnished', 'furnished', 'semi-furnished', None],
        'mainroad': ['yes', 'no', 'yes', 'yes', 'no', 'yes']
    }
    df = pd.DataFrame(data)
    analyzer = CategoricalUnivariateAnalysisStrategy()
    # Perform univariate analysis on 'furnishingstatus'
    analyzer.analyze( df, 'furnishingstatus')

