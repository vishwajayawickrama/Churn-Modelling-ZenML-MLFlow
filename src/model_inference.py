import json
import logging
import os
import sys
import joblib
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.encoders = {} 
        self.load_model()
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("Can't load. File not found.")
        
        self.model = joblib.load(self.model_path)

    def load_encoders(self, encoder_dir):
        for file in os.listdir(encoder_dir):
            feature_name = file.split('_encoder.json')[0]
            with open(os.path.join(encoder_dir, file), 'r') as f:
                self.encoders[feature_name] = json.load(f)

    def preprocess_input(self, data):
        data = pd.DataFrame([data])

        # Encoding
        for col, encoder in self.encoders.items():
            data[col] = data[col].map(encoder)

        # Binning
        binning = CustomBinningStratergy(self.binning_config['credit_score_bins'])
        data = binning.bin_feature(data, 'CreditScore')

        encoder = OrdinalEncodingStratergy(self.encoding_config['ordinal_mappings'])
        data = encoder.encode(data)
        data = data.drop(columns=['RowNumber', 'CustomerId', 'Firstname', 'Lastname'], errors='ignore')

        return data
    
    def predict(self, data):
        pp_data = self.preprocess_input(data)
        y_pred = self.model.predict(pp_data)
        y_prob = float(self.model.predict_proba(pp_data)[0][1])

        y_pred = 'Churn' if y_pred == 1 else 'Retained'
        y_prob = round(y_prob * 100)

        return {
                    "Status": y_pred,
                    "Confidence": f"{y_prob} %"
        }

