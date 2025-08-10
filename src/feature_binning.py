import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureBinningStrategy(ABC):

    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        pass

class CustomBinningStratergy(FeatureBinningStrategy):
    def __init__(self, bin_definition):
        self.bin_definition = bin_definition
    
    def bin_feature(self, df: pd.DataFrame, column: str) ->pd.DataFrame:
        def assign_bin(value):
            if value == 850:
                return 'Excellent'

            for bin_lable, bin_range in self.bin_definition.items():
                if(bin_range) == 1:
                    if bin_range[0] <= value <= bin_range[1]:
                        return bin_lable
                
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return bin_lable
                
                if value > 850:
                    return 'Invalid'
                
                return 'Invalid'
        
        df[f'{column}Bins'] = df[column].apply(assign_bin)
        del df[column]

        return df