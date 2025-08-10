import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class OutlierDetectionStrategy(ABC):

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        outliers = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            outliers[col] = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)

        logging.info('Outliers detected using IQR Method.')

        return outliers
    

class OutlierDetector:
    def __init__(self, strategy : OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df, selected_columns):
        return self.strategy.detect_outliers(df, selected_columns)
    
    def handle_outliers(self, df, selected_columns, method='remove'):
        outlers = self.detect_outliers(df, selected_columns)
        outlier_count = outlers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]

