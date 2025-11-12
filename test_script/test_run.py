"""
Test Script for Volume Forecasting
Runs full pipeline on top 10 SKUs by volume
Generates validation metrics and visualizations
"""

import sys
import os

# Add parent directory to path to import tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Import existing tools (no modifications to production code)
from src.vf_crew.tools.excel_parser import ExcelParserTool
from src.vf_crew.tools.data_validation import DataValidationTool
from src.vf_crew.tools.ts_preparation import TimeSeriesPreparationTool
from src.vf_crew.tools.prophet_model import ProphetModelTool
from src.vf_crew.tools.train_test_split import TrainTestSplitTool
from src.vf_crew.tools.metrics_calculator import MetricsCalculatorTool


class VolumeForecasterTest:
    """Test harness for volume forecasting on top 10 SKUs"""

    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # Results storage
        self.results = []
        self.all_forecasts = []

    def select_top_skus(self, n=10):
        """Select top N SKUs by total volume"""
        print("\n" + "="*60)
        print("STEP 1: Loading Data and Selecting Top SKUs")
        print("="*60)

        # Load Excel file
        print(f"\nLoading Excel file: {self.input_file}")
        parser = ExcelParserTool()
        result = parser._run(self.input_file)

        if not result['success']:
            raise Exception(f"Failed to load Excel: {result.get('error', 'Unknown error')}")

        print(f"[OK] Loaded {result['metadata']['total_time_series']} time series")

        # Calculate total volume per SKU
        time_series_groups = result['time_series_groups']
        sku_volumes = []

        for ts_key, ts_data in time_series_groups.items():
            metadata = ts_data['metadata']
            total_volume = metadata['total_volume']

            sku_volumes.append({
                'ts_key': ts_key,
                'total_volume': total_volume,
                'n_weeks': metadata['n_weeks'],
                'metadata': metadata
            })

        # Sort by volume and get top N
        sku_volumes_df = pd.DataFrame(sku_volumes)
        top_skus = sku_volumes_df.nlargest(n, 'total_volume')

        print(f"\n[OK] Selected top {n} SKUs by volume:\n")
        for idx, row in top_skus.iterrows():
            metadata = row['metadata']
            print(f"  {row.name + 1}. {metadata['Consumer Product'][:50]}...")
            print(f"     Total Volume: {row['total_volume']:,.1f} | Weeks: {row['n_weeks']}")

        return top_skus, time_series_groups

    def process_single_sku(self, rank, ts_key, time_series_groups):
        """Process a single SKU through the full pipeline"""
        print(f"\n[{rank}/10] Processing SKU: {ts_key[:60]}...")

        ts_data = time_series_groups[ts_key]
        metadata = ts_data['metadata']

        try:
            # Prepare data for Prophet
            prep_tool = TimeSeriesPreparationTool()
            valid_series = [{'time_series_key': ts_key, 'metadata': metadata, 'warnings': []}]
            prep_result = prep_tool._run(valid_series, time_series_groups)

            if not prep_result['success']:
                print(f"  [ERROR] Preparation failed: {prep_result.get('error', 'Unknown')}")
                return None

            prepared_data = prep_result['prepared_series'][ts_key]
            full_data = prepared_data['data'].copy()

            # VALIDATION: Train-Test Split (80-20)
            split_tool = TrainTestSplitTool()
            split_result = split_tool._run({'data': full_data}, train_ratio=0.8)

            if not split_result['success']:
                print(f"  [ERROR] Split failed: {split_result.get('error', 'Unknown')}")
                return None

            train_data = split_result['train_data']
            test_data = split_result['test_data']
            split_info = split_result['split_info']

            print(f"  Train: {split_info['train_samples']}w | Test: {split_info['test_samples']}w")

            # Train Prophet on training data
            prophet_tool = ProphetModelTool()
            train_result = prophet_tool._run(
                {'data': train_data},
                forecast_periods=len(test_data),
                use_default_params=True
            )

            if not train_result['success']:
                print(f"  [ERROR] Training failed: {train_result.get('error', 'Unknown')}")
                return None

            # Get predictions on test set
            test_predictions = train_result['future_forecast']

            # Calculate validation metrics
            metrics_tool = MetricsCalculatorTool()
            metrics_result = metrics_tool._run(
                test_data['y'],
                test_predictions['yhat']
            )

            if not metrics_result['success']:
                print(f"  [ERROR] Metrics calculation failed")
                return None

            metrics = metrics_result['metrics']

            print(f"  Validation Metrics:")
            print(f"    MAPE: {metrics['MAPE']:.2f}% | R²: {metrics['R2']:.3f} | RMSE: {metrics['RMSE']:.2f}")

            # FINAL FORECAST: Train on full data
            final_result = prophet_tool._run(
                {'data': full_data},
                forecast_periods=26,  # 6 months
                use_default_params=True
            )

            if not final_result['success']:
                print(f"  [ERROR] Final forecast failed")
                return None

            final_forecast = final_result['future_forecast']
            historical_fit = final_result['historical_fit']

            # Create visualization
            self.create_forecast_chart(
                rank, ts_key, metadata,
                full_data, historical_fit, final_forecast,
                metrics
            )

            print(f"  [OK] Chart saved")

            # Store results
            result_data = {
                'rank': rank,
                'ts_key': ts_key,
                'sku_name': metadata['Consumer Product'],
                'liquid_class': metadata['RTD Liquid Class Filter'],
                'industry_type': metadata['Industry Type'],
                'partition': metadata['Partitions'],
                'total_volume': metadata['total_volume'],
                'train_weeks': split_info['train_samples'],
                'test_weeks': split_info['test_samples'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'Bias': metrics['Bias'],
                'mean_actual': metrics['mean_actual'],
                'mean_predicted': metrics['mean_predicted']
            }

            # Store forecast data
            for _, row in final_forecast.iterrows():
                forecast_row = {
                    'rank': rank,
                    'sku_name': metadata['Consumer Product'],
                    'liquid_class': metadata['RTD Liquid Class Filter'],
                    'industry_type': metadata['Industry Type'],
                    'partition': metadata['Partitions'],
                    'date': row['ds'],
                    'yhat': row['yhat'],
                    'yhat_lower': row['yhat_lower'],
                    'yhat_upper': row['yhat_upper'],
                    'is_forecast': True
                }
                self.all_forecasts.append(forecast_row)

            return result_data

        except Exception as e:
            print(f"  [ERROR] Error: {str(e)}")
            return None

    def create_forecast_chart(self, rank, ts_key, metadata,
                             historical_data, historical_fit, forecast_data, metrics):
        """Create forecast visualization with dotted lines for forecast period"""

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot historical actuals
        ax.plot(
            historical_data['ds'],
            historical_data['y'],
            'ko-',
            label='Actual (Historical)',
            linewidth=1.5,
            markersize=3,
            alpha=0.7
        )

        # Plot historical fitted values (solid line)
        ax.plot(
            historical_fit['ds'],
            historical_fit['yhat'],
            'b-',
            label='Model Fit (Historical)',
            linewidth=2,
            alpha=0.8
        )

        # Plot forecast (DOTTED LINE)
        ax.plot(
            forecast_data['ds'],
            forecast_data['yhat'],
            'b--',  # Dashed/dotted line
            label='Forecast (6 months)',
            linewidth=2.5,
            alpha=0.9
        )

        # Plot confidence interval
        ax.fill_between(
            forecast_data['ds'],
            forecast_data['yhat_lower'],
            forecast_data['yhat_upper'],
            alpha=0.2,
            color='blue',
            label='Confidence Interval (80%)'
        )

        # Add vertical line marking forecast start
        forecast_start = forecast_data['ds'].min()
        ax.axvline(
            x=forecast_start,
            color='red',
            linestyle=':',
            linewidth=2,
            alpha=0.6,
            label='Forecast Start'
        )

        # Formatting
        title = f"Rank #{rank}: {metadata['Consumer Product'][:60]}\n"
        title += f"MAPE: {metrics['MAPE']:.2f}% | R²: {metrics['R2']:.3f} | RMSE: {metrics['RMSE']:.2f}"

        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Volume', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save chart
        chart_path = os.path.join(self.viz_dir, f'sku_{rank:03d}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

    def export_results(self):
        """Export all results to Excel, CSV, and text summary"""
        print("\n" + "="*60)
        print("STEP 3: Exporting Results")
        print("="*60)

        # Export forecasts to Excel
        forecasts_df = pd.DataFrame(self.all_forecasts)
        excel_path = os.path.join(self.output_dir, 'test_forecasts.xlsx')
        forecasts_df.to_excel(excel_path, index=False)
        print(f"\n[OK] Forecasts exported to: {excel_path}")

        # Export metrics to CSV
        metrics_df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, 'validation_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"[OK] Metrics exported to: {csv_path}")

        # Create summary report
        self.create_summary_report(metrics_df)

    def create_summary_report(self, metrics_df):
        """Create text summary report"""
        summary_path = os.path.join(self.output_dir, 'summary_report.txt')

        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Volume Forecasting Test - Top 10 SKUs\n")
            f.write("="*70 + "\n\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total SKUs Processed: {len(self.results)}\n")
            f.write(f"Input File: {self.input_file}\n\n")

            f.write("-"*70 + "\n")
            f.write("Top 10 SKUs by Volume\n")
            f.write("-"*70 + "\n\n")

            for idx, row in metrics_df.iterrows():
                f.write(f"{row['rank']}. {row['sku_name']}\n")
                f.write(f"   Total Volume: {row['total_volume']:,.1f}\n")
                f.write(f"   Partition: {row['partition']}\n\n")

            f.write("-"*70 + "\n")
            f.write("Validation Performance Summary\n")
            f.write("-"*70 + "\n\n")

            avg_mape = metrics_df['MAPE'].mean()
            avg_r2 = metrics_df['R2'].mean()
            avg_rmse = metrics_df['RMSE'].mean()

            f.write(f"Average MAPE: {avg_mape:.2f}%\n")
            f.write(f"Average R²:   {avg_r2:.3f}\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n\n")

            f.write("-"*70 + "\n")
            f.write("Individual SKU Metrics\n")
            f.write("-"*70 + "\n\n")

            f.write(f"{'Rank':<6}{'SKU Name':<40}{'MAPE':<10}{'R²':<10}{'RMSE':<10}\n")
            f.write("-"*70 + "\n")

            for idx, row in metrics_df.iterrows():
                sku_short = row['sku_name'][:37] + "..." if len(row['sku_name']) > 40 else row['sku_name']
                f.write(f"{row['rank']:<6}{sku_short:<40}{row['MAPE']:>8.2f}% {row['R2']:>9.3f} {row['RMSE']:>9.2f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("Output Files\n")
            f.write("="*70 + "\n\n")
            f.write(f"- Forecasts:      {os.path.join(self.output_dir, 'test_forecasts.xlsx')}\n")
            f.write(f"- Metrics:        {os.path.join(self.output_dir, 'validation_metrics.csv')}\n")
            f.write(f"- Visualizations: {self.viz_dir}/ (10 PNG files)\n")
            f.write(f"- This Report:    {summary_path}\n\n")

        print(f"[OK] Summary report saved to: {summary_path}")

    def run(self):
        """Run the full test pipeline"""
        print("\n" + "="*70)
        print(" "*20 + "VOLUME FORECASTING TEST")
        print(" "*20 + "Top 10 SKUs by Volume")
        print("="*70)

        # Step 1: Select top SKUs
        top_skus, time_series_groups = self.select_top_skus(n=10)

        # Step 2: Process each SKU
        print("\n" + "="*60)
        print("STEP 2: Processing SKUs Through Full Pipeline")
        print("="*60)

        for idx, row in top_skus.iterrows():
            result = self.process_single_sku(
                idx + 1,
                row['ts_key'],
                time_series_groups
            )

            if result:
                self.results.append(result)

        # Step 3: Export results
        if self.results:
            self.export_results()

            # Print final summary
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)

            metrics_df = pd.DataFrame(self.results)
            print(f"\nAverage Performance Across {len(self.results)} SKUs:")
            print(f"  - Average MAPE: {metrics_df['MAPE'].mean():.2f}%")
            print(f"  - Average R²:   {metrics_df['R2'].mean():.3f}")
            print(f"  - Average RMSE: {metrics_df['RMSE'].mean():.2f}")

            print(f"\nAll outputs saved to: {self.output_dir}/")
            print(f"  - test_forecasts.xlsx (forecasts for 10 SKUs)")
            print(f"  - validation_metrics.csv (detailed metrics)")
            print(f"  - summary_report.txt (text summary)")
            print(f"  - visualizations/ (10 PNG charts)")

            print("\n" + "="*60)
            print("TEST COMPLETE!")
            print("="*60 + "\n")
        else:
            print("\n[ERROR] No results to export - all SKUs failed processing")


def main():
    """Main entry point"""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.abspath(os.path.join(script_dir, '../data/input/TBS Weekly Volume.xlsx'))
    output_dir = os.path.join(script_dir, 'output')

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please ensure the Excel file is in the correct location.")
        return

    # Run test
    tester = VolumeForecasterTest(input_file, output_dir)
    tester.run()


if __name__ == "__main__":
    main()
