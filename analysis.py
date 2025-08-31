#!/usr/bin/env python3
"""
Simple analysis script for counting flows experiments.
Replaces the overly complex analyze_results.py with clean functionality.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_metrics_files(base_dir: str = "outputs") -> List[Path]:
    """Find all metrics.yaml files in the output directory"""
    base_path = Path(base_dir)
    metrics_files = list(base_path.glob("**/metrics.yaml"))
    logger.info(f"Found {len(metrics_files)} metrics files")
    return metrics_files


def load_experiment_data(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics and config data for a single experiment"""
    try:
        # Load metrics
        with open(metrics_path) as f:
            metrics = yaml.safe_load(f)
        
        # Try to load config from same directory
        config_path = metrics_path.parent / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        # Extract key experiment parameters
        experiment_data = {
            'experiment_path': str(metrics_path.parent),
            'dataset_type': config.get('dataset', {}).get('_target_', 'unknown').split('.')[-1],
            'bridge_type': config.get('bridge', {}).get('_target_', 'unknown').split('.')[-1],
            'model_type': config.get('model', {}).get('_target_', 'unknown').split('.')[-1],
            'seed': config.get('seed', 'unknown'),
            **metrics  # Include all computed metrics
        }
        
        return experiment_data
        
    except Exception as e:
        logger.warning(f"Could not load experiment data from {metrics_path}: {e}")
        return None


def aggregate_results(base_dir: str = "outputs") -> pd.DataFrame:
    """Aggregate all experiment results into a DataFrame"""
    metrics_files = find_metrics_files(base_dir)
    
    if not metrics_files:
        logger.error("No metrics files found!")
        return pd.DataFrame()
    
    all_results = []
    for metrics_path in metrics_files:
        experiment_data = load_experiment_data(metrics_path)
        if experiment_data:
            all_results.append(experiment_data)
    
    if not all_results:
        logger.error("No valid experiment data found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    logger.info(f"Aggregated {len(df)} experiments")
    return df


def create_summary_table(df: pd.DataFrame, output_path: str = "experiment_summary.csv"):
    """Create a clean summary table"""
    if df.empty:
        logger.warning("No data to summarize")
        return
    
    # Select key columns for summary
    summary_cols = ['dataset_type', 'bridge_type', 'model_type', 'seed']
    metric_cols = [col for col in df.columns if col not in summary_cols + ['experiment_path']]
    
    # Round numeric columns
    for col in metric_cols:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(4)
    
    # Save full results
    df.to_csv(output_path, index=False)
    logger.info(f"Saved summary table to {output_path}")
    
    # Print top performers by primary metric
    primary_metrics = ['mean_error', 'mmd_rbf', 'wasserstein_distance']
    primary_metric = None
    for metric in primary_metrics:
        if metric in df.columns:
            primary_metric = metric
            break
    
    if primary_metric:
        print(f"\nTOP 10 EXPERIMENTS BY {primary_metric.upper()}:")
        print("=" * 80)
        top_experiments = df.nsmallest(10, primary_metric)
        display_cols = summary_cols + [primary_metric]
        available_cols = [col for col in display_cols if col in df.columns]
        print(top_experiments[available_cols].to_string(index=False))
        print()


def compare_methods(df: pd.DataFrame, output_path: str = "method_comparison.csv"):
    """Compare different methods across experiments"""
    if df.empty or 'bridge_type' not in df.columns:
        logger.warning("Cannot compare methods - insufficient data")
        return
    
    # Group by method (bridge + model) and compute statistics
    if 'model_type' in df.columns:
        df['method'] = df['bridge_type'] + '_' + df['model_type']
        groupby_cols = ['method']
    else:
        groupby_cols = ['bridge_type']
    
    # Identify metric columns
    metric_cols = [col for col in df.columns 
                   if col not in groupby_cols + ['dataset_type', 'seed', 'experiment_path']]
    
    # Compute aggregated statistics
    comparison_data = []
    for group_name, group_df in df.groupby(groupby_cols):
        result = {'method': group_name if isinstance(group_name, str) else '_'.join(map(str, group_name))}
        result['n_experiments'] = len(group_df)
        
        for metric in metric_cols:
            if metric in group_df.columns and group_df[metric].notna().any():
                values = group_df[metric].dropna()
                if len(values) > 0:
                    result[f'{metric}_mean'] = float(np.mean(values))
                    result[f'{metric}_std'] = float(np.std(values))
                    result[f'{metric}_min'] = float(np.min(values))
                    result[f'{metric}_max'] = float(np.max(values))
        
        comparison_data.append(result)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
    
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved method comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple analysis of counting flows experiments")
    parser.add_argument("--base_dir", default="outputs", help="Base directory to search for results")
    parser.add_argument("--output_dir", default="analysis_results", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing results from: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Aggregate all results
    df = aggregate_results(args.base_dir)
    
    if df.empty:
        print("No experiment results found!")
        return
    
    # Create summary table
    summary_path = output_dir / "experiment_summary.csv"
    create_summary_table(df, str(summary_path))
    
    # Compare methods
    comparison_path = output_dir / "method_comparison.csv"
    compare_methods(df, str(comparison_path))
    
    print(f"\nAnalysis complete!")
    print(f"- Found {len(df)} experiments")
    print(f"- Results saved to {args.output_dir}/")
    print(f"- Summary: {summary_path}")
    print(f"- Method comparison: {comparison_path}")


if __name__ == "__main__":
    main()


