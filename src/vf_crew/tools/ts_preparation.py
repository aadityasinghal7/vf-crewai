"""Time Series Preparation Tool for formatting data for Prophet."""

from typing import Type, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd


class TimeSeriesPreparationInput(BaseModel):
    """Input schema for TimeSeriesPreparationTool."""
    data_file: str = Field(..., description="Path to pickled time series data file from DataValidationTool")


class TimeSeriesPreparationTool(BaseTool):
    name: str = "Time Series Preparation Tool"
    description: str = (
        "Prepares time series data for Prophet model training. "
        "Converts data to Prophet format (ds, y columns), handles edge cases, "
        "and saves ready-to-model data to file."
    )
    args_schema: Type[BaseModel] = TimeSeriesPreparationInput

    def _run(self, data_file: str) -> Dict[str, Any]:
        """
        Prepare time series data for Prophet modeling.

        Args:
            data_file: Path to pickled time series data file from DataValidationTool

        Returns:
            Dictionary containing:
                - data_file: Path to file with Prophet-ready data
                - preparation_summary: Summary statistics
        """
        try:
            import pickle
            import tempfile

            # Load data from file
            with open(data_file, 'rb') as f:
                time_series_groups = pickle.load(f)

            prepared_series = {}
            skipped_series = []

            for ts_key, ts_group in time_series_groups.items():
                try:
                    # Get the original data
                    ts_data = ts_group['data'].copy()
                    metadata = ts_group['metadata']

                    # Rename columns to Prophet format
                    # Prophet requires columns named 'ds' (datestamp) and 'y' (target)
                    ts_data_prophet = ts_data.rename(columns={
                        'Week': 'ds',
                        'Other Sales Volume': 'y'
                    })

                    # Ensure ds is datetime
                    ts_data_prophet['ds'] = pd.to_datetime(ts_data_prophet['ds'])

                    # Sort by date
                    ts_data_prophet = ts_data_prophet.sort_values('ds').reset_index(drop=True)

                    # Check for duplicate dates
                    if ts_data_prophet['ds'].duplicated().any():
                        duplicates = ts_data_prophet[ts_data_prophet['ds'].duplicated(keep=False)]
                        skipped_series.append({
                            'time_series_key': ts_key,
                            'reason': f"Duplicate dates found: {len(duplicates)} duplicates",
                            'metadata': metadata
                        })
                        continue

                    # Store prepared data
                    prepared_series[ts_key] = {
                        'data': ts_data_prophet,
                        'metadata': metadata,
                        'warnings': ts_group.get('warnings', []),
                        'prophet_ready': True,
                        'n_observations': len(ts_data_prophet),
                        'date_range': {
                            'start': ts_data_prophet['ds'].min().strftime('%Y-%m-%d'),
                            'end': ts_data_prophet['ds'].max().strftime('%Y-%m-%d')
                        },
                        'value_stats': {
                            'min': float(ts_data_prophet['y'].min()),
                            'max': float(ts_data_prophet['y'].max()),
                            'mean': float(ts_data_prophet['y'].mean()),
                            'median': float(ts_data_prophet['y'].median()),
                            'std': float(ts_data_prophet['y'].std())
                        }
                    }

                except Exception as e:
                    skipped_series.append({
                        'time_series_key': ts_key,
                        'sku_name': ts_group['metadata'].get('Consumer Product', 'Unknown'),
                        'reason': f"Preparation error: {str(e)}"
                    })

            # Save prepared series to new temp file
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
            pickle.dump(prepared_series, temp_file)
            temp_file.close()

            # Create preparation summary
            total_input = len(time_series_groups)
            preparation_summary = {
                'total_input_series': total_input,
                'successfully_prepared': len(prepared_series),
                'skipped_during_preparation': len(skipped_series),
                'success_rate': len(prepared_series) / total_input * 100 if total_input else 0
            }

            return {
                "success": True,
                "data_file": temp_file.name,
                "skipped_series": skipped_series[:10],  # Only first 10 for summary
                "preparation_summary": preparation_summary,
                "message": f"Prepared {len(prepared_series)} time series for Prophet modeling (skipped {len(skipped_series)}). Data saved to file."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during preparation: {str(e)}",
                "data_file": None,
                "skipped_series": [],
                "preparation_summary": {}
            }
