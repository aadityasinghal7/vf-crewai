"""Excel Parser Tool for reading and parsing volume data."""

from typing import Type, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os


class ExcelParserInput(BaseModel):
    """Input schema for ExcelParserTool."""
    file_path: str = Field(..., description="Path to the Excel file to parse")


class ExcelParserTool(BaseTool):
    name: str = "Excel Parser Tool"
    description: str = (
        "Reads and parses Excel file containing weekly volume data. "
        "Returns structured data grouped by time series combinations. "
        "Expected columns: Week, RTD Liquid Class Filter, Industry Type, "
        "Consumer Product, Partitions, Other Sales Volume."
    )
    args_schema: Type[BaseModel] = ExcelParserInput

    def _run(self, file_path: str) -> Dict[str, Any]:
        """
        Parse Excel file and return structured data.

        Args:
            file_path: Path to Excel file

        Returns:
            Dictionary containing:
                - raw_data: Full DataFrame
                - time_series_groups: Dictionary of DataFrames grouped by product combination
                - metadata: Information about the data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "data": None
                }

            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=0)

            # Validate required columns exist
            required_columns = [
                'Week',
                'RTD Liquid Class Filter',
                'Industry Type',
                'Consumer Product',
                'Partitions',
                'Other Sales Volume'
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing required columns: {missing_columns}",
                    "available_columns": list(df.columns),
                    "data": None
                }

            # Convert Week to datetime if not already
            df['Week'] = pd.to_datetime(df['Week'])

            # Sort by Week
            df = df.sort_values('Week')

            # Group by the combination of categorical columns
            groupby_cols = [
                'RTD Liquid Class Filter',
                'Industry Type',
                'Consumer Product',
                'Partitions'
            ]

            # Create time series groups
            time_series_groups = {}
            for name, group in df.groupby(groupby_cols):
                # Create a unique key for this time series
                ts_key = "|".join(str(x) for x in name)

                # Sort by Week and reset index
                group = group.sort_values('Week').reset_index(drop=True)

                time_series_groups[ts_key] = {
                    'data': group[['Week', 'Other Sales Volume']].copy(),
                    'metadata': {
                        'RTD Liquid Class Filter': name[0],
                        'Industry Type': name[1],
                        'Consumer Product': name[2],
                        'Partitions': name[3],
                        'n_weeks': len(group),
                        'start_date': group['Week'].min().strftime('%Y-%m-%d'),
                        'end_date': group['Week'].max().strftime('%Y-%m-%d'),
                        'total_volume': group['Other Sales Volume'].sum(),
                        'avg_volume': group['Other Sales Volume'].mean(),
                        'has_negatives': (group['Other Sales Volume'] < 0).any(),
                        'has_zeros': (group['Other Sales Volume'] == 0).any()
                    }
                }

            # Create overall metadata
            metadata = {
                'total_records': len(df),
                'total_time_series': len(time_series_groups),
                'date_range': {
                    'start': df['Week'].min().strftime('%Y-%m-%d'),
                    'end': df['Week'].max().strftime('%Y-%m-%d'),
                    'total_weeks': df['Week'].nunique()
                },
                'columns': list(df.columns),
                'unique_values': {
                    'RTD Liquid Class Filter': df['RTD Liquid Class Filter'].nunique(),
                    'Industry Type': df['Industry Type'].nunique(),
                    'Consumer Product': df['Consumer Product'].nunique(),
                    'Partitions': df['Partitions'].nunique()
                }
            }

            # Store the full data in a temporary location to avoid token limits
            # Only return metadata and summary information
            import tempfile
            import pickle

            # Save data to temp file
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
            pickle.dump(time_series_groups, temp_file)
            temp_file.close()

            # Return only summary information
            return {
                "success": True,
                "metadata": metadata,
                "data_file": temp_file.name,  # Path to temporary data file
                "time_series_summary": {
                    ts_key: {
                        'n_weeks': ts_data['metadata']['n_weeks'],
                        'total_volume': ts_data['metadata']['total_volume'],
                        'avg_volume': ts_data['metadata']['avg_volume'],
                        'start_date': ts_data['metadata']['start_date'],
                        'end_date': ts_data['metadata']['end_date']
                    }
                    for ts_key, ts_data in list(time_series_groups.items())[:20]  # Only first 20 for summary
                },
                "message": f"Successfully parsed {len(time_series_groups)} time series from {len(df)} records. Full data saved to temporary file."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing Excel file: {str(e)}",
                "data": None
            }
