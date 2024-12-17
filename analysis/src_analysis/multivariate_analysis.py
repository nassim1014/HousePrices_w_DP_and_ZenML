from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) :
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df):
        # Calculate the correlation matrix
        df_numeric = df.select_dtypes(include=['number'])
        # Compute the correlation matrix
        correlation_matrix = df_numeric.corr()

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    
    def generate_pairplot(self, df):
        # Create a pairplot using seaborn
        df_numeric = df.select_dtypes(include=['number'])
        sns.pairplot(df_numeric)         
        # Title and display the plot
        plt.suptitle("Pairplot", y=1.02)
        plt.show()
            