"""
Feature binning strategies for both pandas and PySpark DataFrames.
Students can compare custom binning implementations and learn PySpark's Bucketizer.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd  # Keep for educational comparison
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureBinningStrategy(ABC):
    """Abstract base class for feature binning strategies."""
    
    ############### PANDAS CODES ###########################
    # @abstractmethod
    # def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
    #     pass
    
    ############### PYSPARK CODES ###########################
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Bin a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with binned feature
        """
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """Custom binning strategy with named bins."""
    
    ############### PANDAS CODES ###########################
    # def __init__(self, bin_definitions):
    #     self.bin_definitions = bin_definitions
    #     logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}")
    #
    # def bin_feature(self, df, column):
    #     logger.info(f"\n{'='*60}")
    #     logger.info(f"FEATURE BINNING - {column.upper()}")
    #     logger.info(f"{'='*60}")
    #     logger.info(f"Starting binning for column: {column}")
    #     initial_unique = df[column].nunique()
    #     value_range = (df[column].min(), df[column].max())
    #     logger.info(f"  Unique values: {initial_unique}, Range: [{value_range[0]:.2f}, {value_range[1]:.2f}]")
    #     
    #     def assign_bin(value):
    #         if value == 850:
    #             return "Excellent"
    #         
    #         for bin_label, bin_range in self.bin_definitions.items():
    #             if len(bin_range) == 2:
    #                 if bin_range[0] <= value <= bin_range[1]:
    #                     return bin_label
    #             elif len(bin_range) == 1:
    #                 if value >= bin_range[0]:
    #                     return bin_label 
    #         
    #         if value > 850:
    #             return "Invalid"
    #         
    #         return "Invalid"
    #     
    #     df[f'{column}Bins'] = df[column].apply(assign_bin)
    #     
    #     # Log binning results
    #     bin_counts = df[f'{column}Bins'].value_counts()
    #     logger.info(f"\nBinning Results:")
    #     for bin_name, count in bin_counts.items():
    #         logger.info(f"  ✓ {bin_name}: {count} ({count/len(df)*100:.2f}%)")
    #     
    #     invalid_count = (df[f'{column}Bins'] == "Invalid").sum()
    #     if invalid_count > 0:
    #         logger.warning(f"  ⚠ Found {invalid_count} invalid values in column '{column}'")
    #         
    #     del df[column]
    #     logger.info(f"✓ Original column '{column}' removed, replaced with '{column}Bins'")
    #     logger.info(f"{'='*60}\n")
    #
    #     return df
    
    ############### PYSPARK CODES ###########################
    def __init__(self, bin_definitions: Dict[str, List[float]], spark: Optional[SparkSession] = None):
        """
        Initialize custom binning strategy.
        
        Args:
            bin_definitions: Dictionary mapping bin names to [min, max] ranges
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.bin_definitions = bin_definitions
        logger.info(f"CustomBinningStrategy initialized with bins: {list(bin_definitions.keys())}")
    
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Apply custom binning to a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with original column replaced by binned version
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE BINNING - {column.upper()} (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting binning for column: {column}")
        
        # Get column statistics
        stats = df.select(
            F.count(F.col(column)).alias('count'),
            F.countDistinct(F.col(column)).alias('unique'),
            F.min(F.col(column)).alias('min'),
            F.max(F.col(column)).alias('max')
        ).collect()[0]
        
        logger.info(f"  Unique values: {stats['unique']}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        # Create binning expression
        bin_column = f"{column}Bins"
        
        # Build CASE WHEN expression
        case_expr = F.when(F.col(column) == 850, "Excellent")
        
        for bin_label, bin_range in self.bin_definitions.items():
            if len(bin_range) == 2:
                case_expr = case_expr.when(
                    (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1]),
                    bin_label
                )
            elif len(bin_range) == 1:
                case_expr = case_expr.when(
                    F.col(column) >= bin_range[0],
                    bin_label
                )
        
        # Handle values > 850 as Invalid
        case_expr = case_expr.when(F.col(column) > 850, "Invalid").otherwise("Invalid")
        
        # Apply binning
        df_binned = df.withColumn(bin_column, case_expr)
        
        # Log binning results
        bin_counts = df_binned.groupBy(bin_column).count().orderBy(F.desc('count')).collect()
        
        logger.info(f"\nBinning Results:")
        total_count = df_binned.count()
        for row in bin_counts:
            bin_name = row[bin_column]
            count = row['count']
            percentage = (count / total_count * 100)
            logger.info(f"  ✓ {bin_name}: {count} ({percentage:.2f}%)")
        
        # Check for invalid values
        invalid_count = df_binned.filter(F.col(bin_column) == "Invalid").count()
        if invalid_count > 0:
            logger.warning(f"  ⚠ Found {invalid_count} invalid values in column '{column}'")
        
        # Drop original column
        df_binned = df_binned.drop(column)
        
        logger.info(f"✓ Original column '{column}' removed, replaced with '{bin_column}'")
        logger.info(f"{'='*60}\n")
        
        return df_binned


############### NEW PYSPARK-SPECIFIC CLASS ###########################
class BucketizerBinningStrategy(FeatureBinningStrategy):
    """Binning strategy using PySpark's Bucketizer."""
    
    def __init__(self, splits: List[float], labels: Optional[List[str]] = None, 
                 handle_invalid: str = "keep", spark: Optional[SparkSession] = None):
        """
        Initialize Bucketizer binning strategy.
        
        Args:
            splits: List of split points for binning (must be monotonically increasing)
            labels: Optional list of bin labels (length should be len(splits) - 1)
            handle_invalid: How to handle values outside splits ("keep", "skip", "error")
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.splits = splits
        self.labels = labels
        self.handle_invalid = handle_invalid
        logger.info(f"BucketizerBinningStrategy initialized with {len(splits)-1} bins")
    
    def bin_feature(self, df: DataFrame, column: str) -> DataFrame:
        """
        Apply Bucketizer binning to a feature column.
        
        Args:
            df: PySpark DataFrame
            column: Column name to bin
            
        Returns:
            DataFrame with binned feature
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BUCKETIZER BINNING - {column.upper()}")
        logger.info(f"{'='*60}")
        
        # Create output column name
        bin_column = f"{column}Bins"
        temp_bin_column = f"{column}_bin_index"
        
        # Create and apply Bucketizer
        bucketizer = Bucketizer(
            splits=self.splits,
            inputCol=column,
            outputCol=temp_bin_column,
            handleInvalid=self.handle_invalid
        )
        
        df_binned = bucketizer.transform(df)
        
        # If labels are provided, map indices to labels
        if self.labels:
            # Create mapping expression
            label_expr = F.when(F.col(temp_bin_column) == 0, self.labels[0])
            for i in range(1, len(self.labels)):
                label_expr = label_expr.when(F.col(temp_bin_column) == i, self.labels[i])
            label_expr = label_expr.otherwise("Unknown")
            
            df_binned = df_binned.withColumn(bin_column, label_expr)
            df_binned = df_binned.drop(temp_bin_column)
        else:
            # Use numeric bin indices
            df_binned = df_binned.withColumnRenamed(temp_bin_column, bin_column)
        
        # Log binning results
        bin_dist = df_binned.groupBy(bin_column).count().orderBy(F.desc('count')).collect()
        total_count = df_binned.count()
        
        logger.info(f"\nBinning Results:")
        for row in bin_dist:
            bin_value = row[bin_column]
            count = row['count']
            percentage = (count / total_count * 100)
            logger.info(f"  ✓ Bin {bin_value}: {count} ({percentage:.2f}%)")
        
        # Drop original column if requested
        df_binned = df_binned.drop(column)
        
        logger.info(f"✓ Binning complete for column '{column}'")
        logger.info(f"{'='*60}\n")
        
        return df_binned


class CreditScoreBinningStrategy(BucketizerBinningStrategy):
    """Specialized binning strategy for credit scores."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize credit score binning with predefined splits and labels."""
        splits = [0, 450, 580, 680, 750, float('inf')]
        labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
        super().__init__(splits=splits, labels=labels, handle_invalid="keep", spark=spark)
        logger.info("CreditScoreBinningStrategy initialized with standard credit score bins")