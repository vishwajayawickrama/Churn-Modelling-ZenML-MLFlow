"""
Data splitting strategies for both pandas and PySpark DataFrames.
Students can compare train-test split implementations between pandas and PySpark.
"""

import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd  # Keep for educational comparison
from sklearn.model_selection import train_test_split  # For pandas implementation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies."""
    
    ############### PANDAS CODES ###########################
    # @abstractmethod
    # def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #     pass
    
    ############### PYSPARK CODES ###########################
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        pass


class SplitType(str, Enum):
    """Enumeration of split types."""
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy."""
    
    ############### PANDAS CODES ###########################
    # def __init__(self, test_size = 0.2):
    #     self.test_size= test_size
    #     logger.info(f"SimpleTrainTestSplitStrategy initialized with test_size={test_size}")
    #
    # def split_data(self, df, target_column):
    #     logger.info(f"\n{'='*60}")
    #     logger.info(f"DATA SPLITTING")
    #     logger.info(f"{'='*60}")
    #     logger.info(f"Starting data splitting with target column: '{target_column}'")
    #     logger.info(f"Total samples: {len(df)}, Features: {len(df.columns) - 1}")
    #     
    #     # Check for missing values
    #     missing_count = df.isnull().sum().sum()
    #     if missing_count > 0:
    #         logger.warning(f"⚠ Dataset contains {missing_count} missing values before splitting")
    #     
    #     Y = df[target_column]
    #     X = df.drop(columns=[target_column])
    #     
    #     # Log target distribution
    #     target_dist = Y.value_counts()
    #     logger.info(f"\nTarget Variable Distribution:")
    #     for value, count in target_dist.items():
    #         logger.info(f"  {value}: {count} ({count/len(Y)*100:.2f}%)")
    #     
    #     # Log feature info
    #     logger.info(f"\nFeature Information:")
    #     logger.info(f"  Number of features: {X.shape[1]}")
    #     logger.info(f"  Feature columns: {list(X.columns)}")
    #     
    #     # Perform split
    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
    #     
    #     # Log split results
    #     logger.info(f"\nSplit Results:")
    #     logger.info(f"  ✓ Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    #     logger.info(f"  ✓ Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    #     
    #     # Log target distribution in train/test sets
    #     train_dist = Y_train.value_counts()
    #     test_dist = Y_test.value_counts()
    #     logger.info(f"\nTarget Distribution in Training Set:")
    #     for value, count in train_dist.items():
    #         logger.info(f"  {value}: {count} ({count/len(Y_train)*100:.2f}%)")
    #     logger.info(f"\nTarget Distribution in Test Set:")
    #     for value, count in test_dist.items():
    #         logger.info(f"  {value}: {count} ({count/len(Y_test)*100:.2f}%)")
    #         
    #     logger.info(f"\n{'='*60}")
    #     logger.info(f"✓ DATA SPLITTING COMPLETE")
    #     logger.info(f"{'='*60}\n")
    #     return X_train, X_test, Y_train, Y_test
    
    ############### PYSPARK CODES ###########################
    def __init__(self, test_size: float = 0.2, random_seed: int = 42, spark: Optional[SparkSession] = None):
        """
        Initialize simple train-test split strategy.
        
        Args:
            test_size: Proportion of data for test set (0-1)
            random_seed: Random seed for reproducibility
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_seed = random_seed
        logger.info(f"SimpleTrainTestSplitStrategy initialized with test_size={test_size}")
    
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Perform simple random train-test split.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA SPLITTING (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting data splitting with target column: '{target_column}'")
        
        # Get total count and columns
        total_samples = df.count()
        all_columns = df.columns
        feature_columns = [col for col in all_columns if col != target_column]
        
        logger.info(f"Total samples: {total_samples}, Features: {len(feature_columns)}")
        
        # Check for missing values
        missing_info = {}
        for col in df.columns:
            missing_count = df.filter(F.col(col).isNull()).count()
            if missing_count > 0:
                missing_info[col] = missing_count
        
        if missing_info:
            total_missing = sum(missing_info.values())
            logger.warning(f"⚠ Dataset contains {total_missing} missing values before splitting")
            for col, count in missing_info.items():
                logger.warning(f"  - {col}: {count} missing values")
        
        # Log target distribution
        logger.info(f"\nTarget Variable Distribution:")
        target_dist = df.groupBy(target_column).count().collect()
        for row in target_dist:
            value = row[target_column]
            count = row['count']
            percentage = (count / total_samples * 100)
            logger.info(f"  {value}: {count} ({percentage:.2f}%)")
        
        # Log feature info
        logger.info(f"\nFeature Information:")
        logger.info(f"  Number of features: {len(feature_columns)}")
        logger.info(f"  Feature columns: {feature_columns}")
        
        # Perform split
        train_df, test_df = df.randomSplit(
            [self.train_size, self.test_size], 
            seed=self.random_seed
        )
        
        # Separate features and target
        X_train = train_df.select(feature_columns)
        Y_train = train_df.select(target_column)
        X_test = test_df.select(feature_columns)
        Y_test = test_df.select(target_column)
        
        # Get counts
        train_count = train_df.count()
        test_count = test_df.count()
        
        # Log split results
        logger.info(f"\nSplit Results:")
        logger.info(f"  ✓ Training set: {train_count} samples ({train_count/total_samples*100:.1f}%)")
        logger.info(f"  ✓ Test set: {test_count} samples ({test_count/total_samples*100:.1f}%)")
        
        # Log target distribution in train/test sets
        logger.info(f"\nTarget Distribution in Training Set:")
        train_target_dist = Y_train.groupBy(target_column).count().collect()
        for row in train_target_dist:
            value = row[target_column]
            count = row['count']
            percentage = (count / train_count * 100)
            logger.info(f"  {value}: {count} ({percentage:.2f}%)")
        
        logger.info(f"\nTarget Distribution in Test Set:")
        test_target_dist = Y_test.groupBy(target_column).count().collect()
        for row in test_target_dist:
            value = row[target_column]
            count = row['count']
            percentage = (count / test_count * 100)
            logger.info(f"  {value}: {count} ({percentage:.2f}%)")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ DATA SPLITTING COMPLETE")
        logger.info(f"{'='*60}\n")
        
        return X_train, X_test, Y_train, Y_test


class StratifiedTrainTestSplitStrategy(DataSplittingStrategy):
    """Stratified train-test split strategy to maintain class distribution."""
    
    def __init__(self, test_size: float = 0.2, random_seed: int = 42, spark: Optional[SparkSession] = None):
        """
        Initialize stratified train-test split strategy.
        
        Args:
            test_size: Proportion of data for test set (0-1)
            random_seed: Random seed for reproducibility
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_seed = random_seed
        logger.info(f"StratifiedTrainTestSplitStrategy initialized with test_size={test_size}")
    
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Perform stratified train-test split to maintain target distribution.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STRATIFIED DATA SPLITTING (PySpark)")
        logger.info(f"{'='*60}")
        
        # Get feature columns
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Initialize empty DataFrames
        train_dfs = []
        test_dfs = []
        
        # Get unique target values
        target_values = df.select(target_column).distinct().collect()
        
        # Split each target class separately
        for row in target_values:
            target_value = row[target_column]
            
            # Filter data for this target value
            target_df = df.filter(F.col(target_column) == target_value)
            target_count = target_df.count()
            
            # Split this subset
            target_train, target_test = target_df.randomSplit(
                [self.train_size, self.test_size],
                seed=self.random_seed
            )
            
            train_dfs.append(target_train)
            test_dfs.append(target_test)
            
            logger.info(f"  Split target={target_value}: {target_count} samples")
        
        # Combine all splits
        train_df = train_dfs[0]
        for df_part in train_dfs[1:]:
            train_df = train_df.union(df_part)
        
        test_df = test_dfs[0]
        for df_part in test_dfs[1:]:
            test_df = test_df.union(df_part)
        
        # Shuffle the combined DataFrames
        train_df = train_df.orderBy(F.rand(seed=self.random_seed))
        test_df = test_df.orderBy(F.rand(seed=self.random_seed))
        
        # Separate features and target
        X_train = train_df.select(feature_columns)
        Y_train = train_df.select(target_column)
        X_test = test_df.select(feature_columns)
        Y_test = test_df.select(target_column)
        
        logger.info(f"\n✓ STRATIFIED SPLITTING COMPLETE")
        logger.info(f"  Training set: {train_df.count()} samples")
        logger.info(f"  Test set: {test_df.count()} samples")
        logger.info(f"{'='*60}\n")
        
        return X_train, X_test, Y_train, Y_test


class DataSplitter:
    """Main data splitter class that uses different strategies."""
    
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialize data splitter with a specific strategy.
        
        Args:
            strategy: DataSplittingStrategy instance
        """
        self.strategy = strategy
        logger.info(f"DataSplitter initialized with strategy: {strategy.__class__.__name__}")
    
    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data using the configured strategy.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        return self.strategy.split_data(df, target_column)