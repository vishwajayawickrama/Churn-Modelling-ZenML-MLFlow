import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import IQROutlierDetection
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStratergy
from data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

def data_pipeline(
                    data_path: str = 'data/raw/ChurnModelling.csv',
                    target_column: str = 'Exited',
                    test_size: float = 0.2,
                    force_rebuild: bool = False
                    ) -> Dict[str, np.ndarray]:
    
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    """
        01. Data Ingestion
    """
    print('Step 1: Data Ingestion')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    if os.path.exists(x_train_path) and \
       os.path.exists(x_test_path) and \
       os.path.exists(y_train_path) and \
       os.path.exists(y_test_path):
        
        X_train =pd.read_csv(x_train_path)
        X_test =pd.read_csv(x_test_path)
        Y_train =pd.read_csv(y_train_path)
        Y_test =pd.read_csv(y_test_path)

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)
    print(f"Loaded Data Shape {df.shape}")

    """
        02. Handling Missing Values
    """
    print('Step 2: Handle Missing Values')

    drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns']) # Dropping Critical Rows in Columns

    age_hanlder = FillMissingValuesStrategy(
                                                relavant_columns='Age',
                                                method='mean'
                                            )
    
    gender_hanlder = FillMissingValuesStrategy(
                                                relavant_columns='Gender',
                                                method='mean',
                                                is_custom_imputer=True,
                                                custom_imputer=GenderImputer()
                                            )
    
    df = drop_handler.handle(df)
    df = age_hanlder.handle(df)
    df = gender_hanlder.handle(df)

    print(f"Data shape after Imputation {df.shape}")

data_pipeline()
