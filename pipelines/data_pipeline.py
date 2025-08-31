import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
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
    print('\nStep 1: Data Ingestion')
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
    
    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)#  exist_ok=True means: If the directory already exists, don’t raise an error — just proceed silently.If it doesn’t exist, create it (and any missing parent directories).
    if not os.path.exists('temp_imputed.csv'):
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        print(f"Loaded Data Shape {df.shape}")
        """
            02. Handling Missong Values
        """
        print('\ntep 2: Handle Missing Values')

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
        df.to_csv('temp_imputed.csv', index=False)
        
    df = pd.read_csv('temp_imputed.csv')
    print(f"Data shape after Imputation {df.shape}")

    """
        03. Handle Outliers
    """
    print('\nStep 3: Handle Outliers')

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
    print(f"Data shape after outleir detected {df.shape}")

    """
        04. Feature Binning
    """
    print('\nStep 4: Feature Binning')

    binning = CustomBinningStratergy(binning_config['credit_score_bins'])
    df = binning.bin_feature(df, 'CreditScore')
    print(f"Data after Feature Binning \n{df.head()}")

    """
        05. Feature Encoding
    """
    print('\nStep 5: Feature Encoding')

    nominal_encoder = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_encoder = OrdinalEncodingStratergy(encoding_config['ordinal_mappings'])

    df = nominal_encoder.encode(df)
    df = ordinal_encoder.encode(df)

    print(f"Data after Fetaure Encoding \n{df.head()}")
    print(f"Data Shape after feature Encoding {df.shape}")

    """
        06. Feature Scaling
    """
    print("\nStep 6: Feature Scaling")
    scaling_stratergy = MinMaxScalingStratergy()
    df = scaling_stratergy.scale(df, scaling_config['columns_to_scale'])
    print(f"Data after Feature Scaling \n{df.head()}")

    """
        07. Post Processing
    """
    print("\nStep 7: Post Processing")
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Firstname', 'Lastname'], axis=1)
    print(f"Data after post processing \n{df.head()}")

    """
        08. Data Splitting
    """
    print("\nStep 8: Data Splitting")
    splitting_stratergy = SimpleTrainTestSplitStratergy(test_size=splitting_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitting_stratergy.split_data(df, 'Exited')

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f"X Train Size : {X_train.shape}")
    print(f"X Test Size : {X_test.shape}")
    print(f"Y Train Size : {Y_train.shape}")
    print(f"Y Test Size : {Y_test.shape}")

if __name__ == "__main__":
    data_pipeline()
