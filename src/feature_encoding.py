import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureEncodingStrategy(ABC):

    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'

class NominalEncodingStrategy(FeatureEncodingStrategy):

    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder_dict = {}
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        for column in self.nominal_columns:
            unique_values = df[column].unique()
            encoder_dict = {value: i for i, value in enumerate(unique_values)}
            self.encoder_dict[column] = encoder_dict

            encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")
            with open(encoder_path, "w") as f:
                json.dump(encoder_dict, f)

            df[column] = df[column].map(encoder_dict)

        return df
    
class OrdinalEncodingStratergy(FeatureEncodingStrategy):

    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings

    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        for column, mapping in self.ordinal_mappings.items():
            df[column] = df[column].map(mapping)
            logging.info(f"Encode ordinal variables with ")
            
        return df



