from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df: pd.DataFrame, feature1 : str, feature2 : str):
        pass

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature1], y=df[feature2], color='blue')
        plt.title(f'Scatter plot between {feature1} and {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
        # Correlation Statistics
        correlation = df[[feature1, feature2]].corr().iloc[0, 1]
        print(f'Correlation between {feature1} and {feature2}: {correlation:.2f}')

class NumericalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):

        # Boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[feature2], y=df[feature1], palette='Set2')
        plt.title(f'Boxplot of {feature1} by {feature2}')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()
        
        # Violin Plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=df[feature2], y=df[feature1], palette='Set2')
        plt.title(f'Violin Plot of {feature1} by {feature2}')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()

class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):
        # Countplot
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[feature1], hue=df[feature2], palette='Set1')
        plt.title(f'Countplot of {feature1} by {feature2}')
        plt.xlabel(feature1)
        plt.ylabel('Count')
        plt.show()

        # Cross-tabulation heatmap
        crosstab = pd.crosstab(df[feature1], df[feature2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d', linewidths=0.5)
        plt.title(f'Heatmap of {feature1} and {feature2}')
        plt.xlabel(feature2)
        plt.ylabel(feature1)
        plt.show()

class BivariateAnalyzer:
    def __init__(self, strategy : BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy : BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df : pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df , feature1, feature2)