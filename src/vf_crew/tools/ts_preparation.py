"""Time Series Preparation Tool for formatting data for Prophet."""

from typing import Type, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd


class TimeSeriesPreparationInput(BaseModel):
    """Input schema for TimeSeriesPreparationTool."""
    valid_series: List[Dict] = Field(..., description="List of valid time series from DataValidationTool")
    time_series_groups: Dict = Field(..., description="Dictionary of time series groups from ExcelParserTool")


class TimeSeriesPreparationTool(BaseTool):
    name: str = "Time Series Preparation Tool"
    description: str = (
        "Prepares time series data for Prophet model training. "
        "Converts data to Prophet format (ds, y columns), handles edge cases, "
        "and returns ready-to-model data."
    )
    args_schema: Type[BaseModel] = TimeSeriesPreparationInput

    def _run(self, valid_series: List[Dict], time_series_groups: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare time series data for Prophet modeling.

        Args:
            valid_series: List of valid time series from validation
            time_series_groups: Dictionary of all time series data

        Returns:
            Dictionary containing:
                - prepared_series: Dictionary of Prophet-ready DataFrames
                - preparation_summary: Summary statistics
        """
        try:
            prepared_series = {}
            skipped_series = []

            for series_info in valid_series:
                ts_key = series_info['time_series_key']

                try:
                    # Get the original data
                    ts_data = time_series_groups[ts_key]['data'].copy()
                    metadata = time_series_groups[ts_key]['metadata']

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
                        'warnings': series_info.get('warnings', []),
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
                        'reason': f"Preparation error: {str(e)}",
                        'metadata': time_series_groups[ts_key]['metadata']
                    })

            # Create preparation summary
            preparation_summary = {
                'total_input_series': len(valid_series),
                'successfully_prepared': len(prepared_series),
                'skipped_during_preparation': len(skipped_series),
                'success_rate': len(prepared_series) / len(valid_series) * 100 if valid_series else 0
            }

            return {
                "success": True,
                "prepared_series": prepared_series,
                "skipped_series": skipped_series,
                "preparation_summary": preparation_summary,
                "message": f"Prepared {len(prepared_series)} time series for Prophet modeling (skipped {len(skipped_series)})"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during preparation: {str(e)}",
                "prepared_series": {},
                "skipped_series": [],
                "preparation_summary": {}
            }
