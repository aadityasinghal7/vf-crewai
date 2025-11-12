"""Data Validation Tool for checking time series quality."""

from typing import Type, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DataValidationInput(BaseModel):
    """Input schema for DataValidationTool."""
    data_file: str = Field(..., description="Path to pickled time series data file from SelectTopSKUsTool")
    min_weeks_required: int = Field(default=10, description="Minimum weeks of data required for a valid time series")


class DataValidationTool(BaseTool):
    name: str = "Data Validation Tool"
    description: str = (
        "Validates time series data quality. Checks for sufficient history, "
        "missing values, negative values, and other data quality issues. "
        "Takes a data_file path and returns validated series data file path."
    )
    args_schema: Type[BaseModel] = DataValidationInput

    def _run(self, data_file: str, min_weeks_required: int = 10) -> Dict[str, Any]:
        """
        Validate time series data quality.

        Args:
            data_file: Path to pickled time series data file
            min_weeks_required: Minimum number of weeks required for a valid series

        Returns:
            Dictionary containing:
                - data_file: Path to file with valid series only
                - invalid_series: List of flagged time series with reasons (summary only)
                - validation_summary: Summary statistics
        """
        try:
            import pickle
            import tempfile

            # Load data from file
            with open(data_file, 'rb') as f:
                time_series_groups = pickle.load(f)

            valid_series = []
            valid_series_data = {}
            invalid_series = []

            # Iterate through each time series
            for ts_key, ts_data in time_series_groups.items():
                metadata = ts_data['metadata']
                data = ts_data['data']

                # Track validation issues
                issues = []

                # Check 1: Insufficient history
                if metadata['n_weeks'] < min_weeks_required:
                    issues.append(f"Insufficient history: {metadata['n_weeks']} weeks (minimum: {min_weeks_required})")

                # Check 2: All zeros
                if data['Other Sales Volume'].sum() == 0:
                    issues.append("All values are zero")

                # Check 3: More than 50% zeros
                zero_pct = (data['Other Sales Volume'] == 0).sum() / len(data) * 100
                if zero_pct > 50:
                    issues.append(f"Too many zeros: {zero_pct:.1f}% of data")

                # Check 4: Missing values
                if data['Other Sales Volume'].isna().any():
                    missing_count = data['Other Sales Volume'].isna().sum()
                    issues.append(f"Missing values: {missing_count} records")

                # Check 5: Check for constant values (no variance)
                if data['Other Sales Volume'].std() == 0:
                    issues.append("No variance in values (constant series)")

                # Information: Flag negative values (but don't invalidate)
                if metadata['has_negatives']:
                    negative_count = (data['Other Sales Volume'] < 0).sum()
                    issues.append(f"Info: Contains {negative_count} negative values (likely returns)")

                # Decide if series is valid or invalid
                # Only the first 5 checks are hard failures
                critical_issues = [issue for issue in issues if not issue.startswith("Info:")]

                if critical_issues:
                    invalid_series.append({
                        'time_series_key': ts_key,
                        'sku_name': metadata.get('Consumer Product', 'Unknown'),
                        'issues': issues,
                        'critical': True
                    })
                else:
                    # Valid series (with or without warnings)
                    valid_series.append({
                        'time_series_key': ts_key,
                        'sku_name': metadata.get('Consumer Product', 'Unknown'),
                        'warnings': issues if issues else []
                    })
                    # Keep the full data for valid series
                    valid_series_data[ts_key] = ts_data

            # Save valid series to new temp file
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
            pickle.dump(valid_series_data, temp_file)
            temp_file.close()

            # Create validation summary
            validation_summary = {
                'total_time_series': len(time_series_groups),
                'valid_series_count': len(valid_series),
                'invalid_series_count': len(invalid_series),
                'validation_rate': len(valid_series) / len(time_series_groups) * 100 if time_series_groups else 0,
                'min_weeks_threshold': min_weeks_required,
                'invalid_reasons': {}
            }

            # Count invalid reasons
            for invalid_ts in invalid_series:
                for issue in invalid_ts['issues']:
                    # Extract main reason (before the colon)
                    reason = issue.split(':')[0]
                    validation_summary['invalid_reasons'][reason] = \
                        validation_summary['invalid_reasons'].get(reason, 0) + 1

            return {
                "success": True,
                "data_file": temp_file.name,
                "valid_series_summary": valid_series[:10],  # Only first 10 for summary
                "invalid_series": invalid_series[:10],  # Only first 10 for summary
                "validation_summary": validation_summary,
                "message": f"Validation complete: {len(valid_series)} valid, {len(invalid_series)} invalid out of {len(time_series_groups)} total. Valid data saved to file."
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error during validation: {str(e)}",
                "data_file": None,
                "valid_series_summary": [],
                "invalid_series": [],
                "validation_summary": {}
            }
