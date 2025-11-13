"""Train-Test Split Tool for validation."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import math


class TrainTestSplitInput(BaseModel):
    """Input schema for TrainTestSplitTool."""
    data_file: str = Field(..., description="Path to pickled Prophet-formatted time series data file")
    train_ratio: float = Field(default=0.8, description="Proportion of data for training (default: 0.8)")


class TrainTestSplitTool(BaseTool):
    name: str = "Train-Test Split Tool"
    description: str = (
        "Performs temporal train-test split on time series data. "
        "Splits data chronologically to avoid data leakage. "
        "Takes a data_file path from TimeSeriesPreparationTool and returns train_data_file and test_data_file paths. "
        "Default: 80% training, 20% testing."
    )
    args_schema: Type[BaseModel] = TrainTestSplitInput

    def _run(
        self,
        data_file: str,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Split time series data temporally for validation.

        Args:
            data_file: Path to pickled Prophet-formatted time series data file
            train_ratio: Proportion of data for training (0 to 1)

        Returns:
            Dictionary containing:
                - train_data_file: Path to pickled file with training data
                - test_data_file: Path to pickled file with testing data
                - split_summaries: Summary of splits for each series
                - success: Whether split was successful
        """
        try:
            import pickle
            import tempfile

            # Load data from pickle file
            with open(data_file, 'rb') as f:
                time_series_groups = pickle.load(f)

            # Process each time series and perform train-test split
            train_splits = {}
            test_splits = {}
            split_summaries = []

            for ts_key, ts_data in time_series_groups.items():
                # Extract the DataFrame
                if isinstance(ts_data, dict) and 'data' in ts_data:
                    df = ts_data['data'].copy()
                    metadata = ts_data.get('metadata', {})
                else:
                    df = ts_data.copy()
                    metadata = {}

                # Validate input
                if not isinstance(df, pd.DataFrame):
                    split_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Invalid data format'
                    })
                    continue

                if 'ds' not in df.columns or 'y' not in df.columns:
                    split_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Missing ds or y columns'
                    })
                    continue

                # Sort by date
                df = df.sort_values('ds').reset_index(drop=True)

                # Calculate split point
                n_total = len(df)
                n_train = math.floor(n_total * train_ratio)

                # Ensure at least some test data
                if n_train >= n_total - 1:
                    n_train = n_total - 2  # Leave at least 2 points for testing

                # Ensure minimum training data
                if n_train < 10:
                    split_summaries.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': f'Insufficient training data after split: {n_train} points (minimum: 10)'
                    })
                    continue

                # Perform split
                train_data = df.iloc[:n_train].copy()
                test_data = df.iloc[n_train:].copy()

                # Store train and test splits with metadata
                train_splits[ts_key] = {
                    'data': train_data,
                    'metadata': metadata
                }

                test_splits[ts_key] = {
                    'data': test_data,
                    'metadata': metadata
                }

                split_summaries.append({
                    'ts_key': ts_key,
                    'sku_name': metadata.get('Consumer Product', 'Unknown'),
                    'status': 'success',
                    'total_samples': n_total,
                    'train_samples': n_train,
                    'test_samples': len(test_data),
                    'train_ratio_actual': round(n_train / n_total, 3),
                    'test_ratio_actual': round(len(test_data) / n_total, 3),
                    'split_date': train_data['ds'].max().strftime('%Y-%m-%d')
                })

            # Save train and test splits to separate temp files
            train_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='_train.pkl')
            pickle.dump(train_splits, train_file)
            train_file.close()

            test_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='_test.pkl')
            pickle.dump(test_splits, test_file)
            test_file.close()

            # Calculate max test samples for Prophet forecast_periods
            successful_splits = [s for s in split_summaries if s['status'] == 'success']
            max_test_samples = max([s['test_samples'] for s in successful_splits]) if successful_splits else 0

            return {
                "success": True,
                "train_data_file": train_file.name,
                "test_data_file": test_file.name,
                "split_summaries": split_summaries[:10],  # First 10 for summary
                "max_test_samples": max_test_samples,  # For Prophet forecast_periods during validation
                "total_splits": len(train_splits),
                "failed_splits": len([s for s in split_summaries if s['status'] == 'failed']),
                "message": f"Successfully split {len(train_splits)} time series (train_ratio={train_ratio}). Max test samples: {max_test_samples}. For validation, use forecast_periods={max_test_samples} with ProphetModelTool."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during train-test split: {str(e)}",
                "train_data_file": None,
                "test_data_file": None,
                "split_summaries": [],
                "total_splits": 0,
                "failed_splits": 0
            }
