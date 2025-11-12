"""Metrics Calculator Tool for validation assessment."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculatorInput(BaseModel):
    """Input schema for MetricsCalculatorTool."""
    test_data_file: str = Field(..., description="Path to pickled file with test data (actual values)")
    forecast_data_file: str = Field(..., description="Path to pickled file with forecast data (predicted values)")


class MetricsCalculatorTool(BaseTool):
    name: str = "Metrics Calculator Tool"
    description: str = (
        "Calculates comprehensive validation metrics for forecast accuracy. "
        "Computes MAE, RMSE, MAPE, R², and forecast bias. "
        "Takes test_data_file and forecast_data_file paths and returns metrics for all time series. "
        "Handles edge cases and provides detailed metric explanations."
    )
    args_schema: Type[BaseModel] = MetricsCalculatorInput

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        Handles zero values by masking them out.
        """
        # Mask zero values to avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.nan  # Cannot calculate MAPE if all actuals are zero

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def _calculate_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate forecast bias (mean error).
        Positive bias = over-forecasting, Negative bias = under-forecasting.
        """
        return np.mean(y_pred - y_true)

    def _run(
        self,
        test_data_file: str,
        forecast_data_file: str
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive validation metrics for all time series.

        Args:
            test_data_file: Path to pickled file with test data (actual values)
            forecast_data_file: Path to pickled file with forecast data (predicted values)

        Returns:
            Dictionary containing:
                - metrics_results: List of metrics for each time series
                - summary_stats: Aggregate statistics across all time series
                - success: Whether calculation was successful
        """
        try:
            import pickle

            # Load test data and forecast data from pickle files
            with open(test_data_file, 'rb') as f:
                test_data_groups = pickle.load(f)

            with open(forecast_data_file, 'rb') as f:
                forecast_data_groups = pickle.load(f)

            # Process each time series and calculate metrics
            metrics_results = []

            for ts_key in test_data_groups.keys():
                if ts_key not in forecast_data_groups:
                    metrics_results.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Forecast data not found for this time series'
                    })
                    continue

                # Extract test data
                test_data_dict = test_data_groups[ts_key]
                test_df = test_data_dict['data'] if isinstance(test_data_dict, dict) and 'data' in test_data_dict else test_data_dict
                metadata = test_data_dict.get('metadata', {}) if isinstance(test_data_dict, dict) else {}

                # Extract forecast data
                forecast_dict = forecast_data_groups[ts_key]
                if 'historical_fit' in forecast_dict:
                    # Prophet output format - we need the historical fit for the test period
                    forecast_full = forecast_dict['forecast_full']
                    # Match dates with test data
                    test_dates = test_df['ds'].values
                    forecast_for_test = forecast_full[forecast_full['ds'].isin(test_dates)]

                    if len(forecast_for_test) == 0:
                        metrics_results.append({
                            'ts_key': ts_key,
                            'status': 'failed',
                            'error': 'No matching forecast dates for test period'
                        })
                        continue

                    # Align by date
                    test_df_sorted = test_df.sort_values('ds').reset_index(drop=True)
                    forecast_sorted = forecast_for_test.sort_values('ds').reset_index(drop=True)

                    y_true = test_df_sorted['y'].values
                    y_pred = forecast_sorted['yhat'].values
                else:
                    # Fallback format
                    metrics_results.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Unexpected forecast data format'
                    })
                    continue

                # Validate lengths match
                if len(y_true) != len(y_pred):
                    metrics_results.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': f'Length mismatch: actual={len(y_true)}, predicted={len(y_pred)}'
                    })
                    continue

                # Check for sufficient data
                if len(y_true) < 2:
                    metrics_results.append({
                        'ts_key': ts_key,
                        'status': 'failed',
                        'error': 'Insufficient data points for metric calculation (minimum: 2)'
                    })
                    continue

                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = self._calculate_mape(y_true, y_pred)
                bias = self._calculate_bias(y_true, y_pred)

                # R² score (handle edge case where variance is 0)
                try:
                    r2 = r2_score(y_true, y_pred)
                except:
                    r2 = np.nan

                # Additional useful metrics
                mean_actual = np.mean(y_true)
                mean_predicted = np.mean(y_pred)

                # Calculate relative metrics (as percentage of mean)
                mae_percentage = (mae / mean_actual * 100) if mean_actual != 0 else np.nan
                rmse_percentage = (rmse / mean_actual * 100) if mean_actual != 0 else np.nan

                metrics_results.append({
                    'ts_key': ts_key,
                    'sku_name': metadata.get('Consumer Product', 'Unknown'),
                    'status': 'success',
                    'metrics': {
                        'MAE': float(mae),
                        'RMSE': float(rmse),
                        'MAPE': float(mape) if not np.isnan(mape) else None,
                        'R2': float(r2) if not np.isnan(r2) else None,
                        'Bias': float(bias),
                        'MAE_percent': float(mae_percentage) if not np.isnan(mae_percentage) else None,
                        'RMSE_percent': float(rmse_percentage) if not np.isnan(rmse_percentage) else None,
                        'mean_actual': float(mean_actual),
                        'mean_predicted': float(mean_predicted),
                        'n_samples': int(len(y_true))
                    }
                })

            # Calculate summary statistics
            successful_metrics = [m for m in metrics_results if m['status'] == 'success']

            if successful_metrics:
                summary_stats = {
                    'avg_MAE': float(np.mean([m['metrics']['MAE'] for m in successful_metrics])),
                    'avg_RMSE': float(np.mean([m['metrics']['RMSE'] for m in successful_metrics])),
                    'avg_MAPE': float(np.mean([m['metrics']['MAPE'] for m in successful_metrics if m['metrics']['MAPE'] is not None])) if any(m['metrics']['MAPE'] is not None for m in successful_metrics) else None,
                    'avg_R2': float(np.mean([m['metrics']['R2'] for m in successful_metrics if m['metrics']['R2'] is not None])) if any(m['metrics']['R2'] is not None for m in successful_metrics) else None,
                    'total_series': len(metrics_results),
                    'successful_series': len(successful_metrics),
                    'failed_series': len(metrics_results) - len(successful_metrics)
                }
            else:
                summary_stats = {
                    'total_series': len(metrics_results),
                    'successful_series': 0,
                    'failed_series': len(metrics_results)
                }

            return {
                "success": True,
                "metrics_results": metrics_results[:10],  # First 10 for summary
                "summary_stats": summary_stats,
                "message": f"Calculated metrics for {len(successful_metrics)} time series. Avg MAE: {summary_stats.get('avg_MAE', 'N/A')}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error calculating metrics: {str(e)}",
                "metrics_results": [],
                "summary_stats": {}
            }

    def _interpret_r2(self, r2: float) -> str:
        """Interpret R² score quality."""
        if r2 >= 0.9:
            return 'excellent'
        elif r2 >= 0.7:
            return 'good'
        elif r2 >= 0.5:
            return 'moderate'
        elif r2 >= 0:
            return 'poor'
        else:
            return 'very_poor'
