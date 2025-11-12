"""Excel Export Tool for writing results with visualizations."""

from typing import Type, Dict, Any, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import os
from datetime import datetime


class ExcelExportInput(BaseModel):
    """Input schema for ExcelExportTool."""
    forecasts_file: str = Field(..., description="Path to pickled file containing forecast data from ProphetModelTool")
    validation_results: Optional[Dict] = Field(None, description="Validation results dict from validation task (contains metrics_results)")
    data_prep_results: Optional[Dict] = Field(None, description="Data preparation results dict (contains invalid_series)")
    output_path: str = Field(..., description="Path to save the Excel file")
    summary_stats: Optional[Dict] = Field(None, description="Optional summary statistics")


class ExcelExportTool(BaseTool):
    name: str = "Excel Export Tool"
    description: str = (
        "Exports forecast results, validation metrics, and visualizations to Excel. "
        "Takes forecasts_file path and validation/data prep results dicts. "
        "Creates multiple sheets: forecasts, validation metrics, flagged series, and summary. "
        "Includes formatting and professional layout."
    )
    args_schema: Type[BaseModel] = ExcelExportInput

    def _run(
        self,
        forecasts_file: str,
        validation_results: Optional[Dict] = None,
        data_prep_results: Optional[Dict] = None,
        output_path: str = None,
        summary_stats: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Export results to Excel file.

        Args:
            forecasts_file: Path to pickled file with forecast data
            validation_results: Dict containing validation metrics_results
            data_prep_results: Dict containing invalid_series list
            output_path: Path to save Excel file
            summary_stats: Optional summary statistics

        Returns:
            Dictionary with export status and file info
        """
        try:
            import pickle

            # Load forecast data from pickle file
            with open(forecasts_file, 'rb') as f:
                forecasts_data = pickle.load(f)

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4472C4',
                    'font_color': 'white',
                    'border': 1
                })
                number_format = workbook.add_format({'num_format': '#,##0.00'})
                date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})

                # Sheet 1: Forecasts (Future Forecasts Only)
                forecasts_list = []
                for ts_key, ts_forecast in forecasts_data.items():
                    # Extract metadata
                    metadata = ts_forecast.get('metadata', {})

                    # Get future forecast only
                    future_forecast_df = ts_forecast.get('future_forecast', pd.DataFrame())

                    if not future_forecast_df.empty:
                        forecast_df = future_forecast_df.copy()

                        # Add metadata columns at the start
                        forecast_df.insert(0, 'Partition', metadata.get('Partitions', ''))
                        forecast_df.insert(0, 'Consumer_Product', metadata.get('Consumer Product', ''))
                        forecast_df.insert(0, 'Industry_Type', metadata.get('Industry Type', ''))
                        forecast_df.insert(0, 'Liquid_Class', metadata.get('RTD Liquid Class Filter', ''))
                        forecast_df.insert(0, 'Time_Series_Key', ts_key)

                        # Rename forecast columns for clarity
                        forecast_df = forecast_df.rename(columns={
                            'ds': 'Forecast_Date',
                            'yhat': 'Forecasted_Volume',
                            'yhat_lower': 'Lower_Bound',
                            'yhat_upper': 'Upper_Bound'
                        })

                        forecasts_list.append(forecast_df)

                if forecasts_list:
                    forecasts_combined = pd.concat(forecasts_list, ignore_index=True)

                    # Select and order columns
                    cols = ['Time_Series_Key', 'Liquid_Class', 'Industry_Type', 'Consumer_Product', 'Partition',
                            'Forecast_Date', 'Forecasted_Volume', 'Lower_Bound', 'Upper_Bound']
                    forecasts_combined = forecasts_combined[[c for c in cols if c in forecasts_combined.columns]]

                    forecasts_combined.to_excel(writer, sheet_name='Forecasts', index=False)

                    # Format the sheet
                    worksheet = writer.sheets['Forecasts']
                    for col_num, value in enumerate(forecasts_combined.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Auto-adjust column widths
                    worksheet.set_column('A:A', 40)  # Time_Series_Key
                    worksheet.set_column('B:E', 20)  # Metadata columns
                    worksheet.set_column('F:F', 15)  # Date
                    worksheet.set_column('G:I', 18)  # Numeric columns

                # Sheet 2: Validation Metrics
                validation_metrics = []
                if validation_results and 'metrics_results' in validation_results:
                    validation_metrics = validation_results['metrics_results']

                if validation_metrics:
                    # Flatten metrics for export
                    metrics_list = []
                    for item in validation_metrics:
                        if item.get('status') == 'success':
                            metrics_dict = {
                                'Time_Series_Key': item.get('ts_key', ''),
                                'SKU_Name': item.get('sku_name', ''),
                                'Status': item.get('status', ''),
                            }
                            # Add all metrics
                            for metric_name, metric_value in item.get('metrics', {}).items():
                                metrics_dict[metric_name] = metric_value

                            metrics_list.append(metrics_dict)
                        else:
                            # Failed metric calculation
                            metrics_list.append({
                                'Time_Series_Key': item.get('ts_key', ''),
                                'SKU_Name': item.get('sku_name', ''),
                                'Status': item.get('status', ''),
                                'Error': item.get('error', '')
                            })

                    if metrics_list:
                        metrics_df = pd.DataFrame(metrics_list)
                        metrics_df.to_excel(writer, sheet_name='Validation_Metrics', index=False)

                        worksheet = writer.sheets['Validation_Metrics']
                        for col_num, value in enumerate(metrics_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column('A:A', 40)  # Time_Series_Key
                        worksheet.set_column('B:B', 25)  # SKU_Name

                # Sheet 3: Flagged/Skipped Series
                invalid_series = []
                if data_prep_results and 'invalid_series' in data_prep_results:
                    invalid_series = data_prep_results['invalid_series']

                if invalid_series:
                    flagged_list = []
                    for item in invalid_series:
                        time_series_key = item.get('time_series_key', '')
                        sku_name = item.get('sku_name', 'Unknown')
                        issues = item.get('issues', [])

                        flagged_list.append({
                            'Time_Series_Key': time_series_key,
                            'SKU_Name': sku_name,
                            'Issues': ' | '.join(issues) if isinstance(issues, list) else str(issues),
                            'Critical': item.get('critical', False)
                        })

                    if flagged_list:
                        flagged_df = pd.DataFrame(flagged_list)
                        flagged_df.to_excel(writer, sheet_name='Flagged_Series', index=False)

                        worksheet = writer.sheets['Flagged_Series']
                        for col_num, value in enumerate(flagged_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column('A:A', 40)  # Time_Series_Key
                        worksheet.set_column('B:B', 25)  # SKU_Name
                        worksheet.set_column('C:C', 60)  # Issues

                # Sheet 4: Summary
                summary_data = []
                summary_data.append(['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                summary_data.append(['Total Forecasts Generated', len(forecasts_list)])
                summary_data.append(['Total Validation Metrics', len(validation_metrics)])
                summary_data.append(['Total Flagged Series', len(invalid_series)])

                # Add validation summary stats
                if validation_results and 'summary_stats' in validation_results:
                    val_summary = validation_results['summary_stats']
                    summary_data.append(['', ''])
                    summary_data.append(['=== Validation Summary ===', ''])
                    for key, value in val_summary.items():
                        summary_data.append([str(key), str(value)])

                # Add data prep summary stats
                if data_prep_results and 'summary' in data_prep_results:
                    prep_summary = data_prep_results['summary']
                    summary_data.append(['', ''])
                    summary_data.append(['=== Data Preparation Summary ===', ''])
                    for key, value in prep_summary.items():
                        summary_data.append([str(key), str(value)])

                # Add additional summary stats if provided
                if summary_stats:
                    summary_data.append(['', ''])
                    summary_data.append(['=== Additional Statistics ===', ''])
                    for key, value in summary_stats.items():
                        summary_data.append([str(key), str(value)])

                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                worksheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                worksheet.set_column('A:A', 35)
                worksheet.set_column('B:B', 45)

            # Get file size
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)

            return {
                "success": True,
                "output_path": output_path,
                "sheets_created": ['Forecasts', 'Validation_Metrics', 'Flagged_Series', 'Summary'],
                "forecasts_count": len(forecasts_list),
                "metrics_count": len(validation_metrics),
                "flagged_count": len(invalid_series),
                "file_size_mb": round(file_size_mb, 2),
                "message": f"Successfully exported results to {output_path} ({file_size_mb:.2f} MB)"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Excel export failed: {str(e)}",
                "output_path": None,
                "sheets_created": [],
                "forecasts_count": 0,
                "metrics_count": 0,
                "flagged_count": 0
            }
