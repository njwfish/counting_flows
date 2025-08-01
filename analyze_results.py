#!/usr/bin/env python3
"""
Comprehensive results analysis for counting flows experiments.
Aggregates results from multirun sweeps and creates comparison tables.
"""

import os
import pickle
import pandas as pd
import numpy as np
import yaml
import scipy.stats
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_eval_files(base_dir: str = "outputs") -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Find all eval_data.pkl files and their corresponding configs.
    
    Returns:
        List of (eval_path, config_dict) tuples
    """
    base_path = Path(base_dir)
    eval_files = []
    
    # Handle both single runs and multirun outputs
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # Look for eval_data.pkl files
        if "eval_data.pkl" in files:
            eval_path = root_path / "eval_data.pkl"
            config_path = root_path / "config.yaml"
            
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    eval_files.append((eval_path, config))
                    logger.debug(f"Found eval file: {eval_path}")
                except Exception as e:
                    logger.warning(f"Could not load config from {config_path}: {e}")
    
    logger.info(f"Found {len(eval_files)} evaluation files")
    return eval_files


def extract_experiment_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ALL experimental parameters from config, excluding only seed."""
    import pandas as pd  # For NaN handling
    params = {}
    
    def flatten_config(cfg: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Recursively flatten configuration into dot-separated keys."""
        flattened = {}
        
        for key, value in cfg.items():
            # Skip seed entirely
            if key == 'seed':
                continue
                
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Special handling for nested configs
                if '_target_' in value:
                    # For instantiated objects, use the class name and flatten params
                    flattened[f"{full_key}._target_"] = value['_target_'].split('.')[-1]
                    # Add all other parameters in this dict
                    for sub_key, sub_value in value.items():
                        if sub_key != '_target_':
                            flattened[f"{full_key}.{sub_key}"] = _serialize_value(sub_value)
                else:
                    # Regular nested dict - flatten recursively
                    flattened.update(flatten_config(value, full_key))
            else:
                # Simple value
                flattened[full_key] = _serialize_value(value)
        
        return flattened
    
    def _serialize_value(value: Any) -> str:
        """Convert any value to a string for grouping purposes with proper normalization."""
        import pandas as pd  # Import here for pd.isna()
        
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 'None'
        elif isinstance(value, (int, float)):
            # Normalize numeric values - remove trailing zeros and handle precision
            if isinstance(value, float) and value.is_integer():
                return str(int(value))  # 8.0 -> "8"
            else:
                return f"{value:g}"  # Remove trailing zeros: 8.0 -> "8", 8.5 -> "8.5"
        elif isinstance(value, str):
            # Handle string representations of numbers
            try:
                # Try to parse as number and normalize
                if '.' in value:
                    num_val = float(value)
                    if num_val.is_integer():
                        return str(int(num_val))  # "8.0" -> "8"
                    else:
                        return f"{num_val:g}"  # "8.50" -> "8.5"
                else:
                    # Try integer
                    return str(int(value))  # "8" -> "8"
            except (ValueError, TypeError):
                # Not a number, return as-is but stripped
                return str(value).strip()
        elif isinstance(value, (list, tuple)):
            # For lists/tuples, convert each element to string and sort
            try:
                str_items = []
                for item in value:
                    if item is None or (isinstance(item, float) and pd.isna(item)):
                        str_items.append('None')
                    elif isinstance(item, (int, float)):
                        if isinstance(item, float) and item.is_integer():
                            str_items.append(str(int(item)))
                        else:
                            str_items.append(f"{item:g}")
                    else:
                        str_items.append(str(item).strip())
                
                # Try to sort numerically if all are numbers
                try:
                    numeric_items = [float(x) if x != 'None' else -float('inf') for x in str_items]
                    sorted_items = [str_items[i] for i in sorted(range(len(str_items)), key=lambda i: numeric_items[i])]
                    return str(sorted_items)
                except (ValueError, TypeError):
                    # Fall back to alphabetical sort
                    return str(sorted(str_items))
            except (TypeError, ValueError):
                return str(list(value))
        elif isinstance(value, dict):
            # For dicts, normalize each value
            try:
                normalized_items = []
                for k, v in value.items():
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        normalized_items.append((k, 'None'))
                    elif isinstance(v, (int, float)):
                        if isinstance(v, float) and v.is_integer():
                            normalized_items.append((k, str(int(v))))
                        else:
                            normalized_items.append((k, f"{v:g}"))
                    else:
                        normalized_items.append((k, str(v).strip()))
                
                return str(sorted(normalized_items))
            except (TypeError, ValueError):
                return str(dict(value))
        else:
            return str(value).strip()
    
    # Flatten the entire config
    flattened = flatten_config(config)
    
    # Keep seed separate for reference (not for grouping)
    params['seed'] = config.get('seed', 'unknown')
    
    # Add core identifiers for easy reading
    params['dataset_type'] = config.get('dataset', {}).get('_target_', 'unknown').split('.')[-1]
    params['bridge_type'] = config.get('bridge', {}).get('_target_', 'unknown').split('.')[-1]
    params['model_type'] = config.get('model', {}).get('_target_', 'unknown').split('.')[-1]
    params['architecture_type'] = config.get('architecture', {}).get('_target_', 'unknown').split('.')[-1]
    
    # Add all flattened parameters for complete differentiation
    params.update(flattened)
    
    # Post-process to handle missing values consistently
    # Convert empty strings and NaN to 'None' for consistent grouping
    for key, value in params.items():
        if key == 'seed':  # Skip seed
            continue
        if value == '' or value == 'nan' or (isinstance(value, float) and pd.isna(value)):
            params[key] = 'None'
    
    return params


def load_evaluation_data(eval_path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation data from pickle file."""
    try:
        with open(eval_path, 'rb') as f:
            eval_data = pickle.load(f)
        return eval_data
    except Exception as e:
        logger.error(f"Could not load eval data from {eval_path}: {e}")
        return None


def compute_comprehensive_metrics(eval_data: Dict[str, Any]) -> Dict[str, float]:
    """Compute comprehensive distributional metrics for generative models."""
    x0_target = eval_data['x0_target']
    x0_generated = eval_data['x0_generated']
    
    metrics = {}
    
    # Ensure both arrays have the same shape for comparison
    min_samples = min(len(x0_target), len(x0_generated))
    x0_target = x0_target[:min_samples]
    x0_generated = x0_generated[:min_samples]
    
    # ====== MARGINAL MOMENT METRICS ======
    target_mean = np.mean(x0_target, axis=0)
    gen_mean = np.mean(x0_generated, axis=0)
    target_var = np.var(x0_target, axis=0)
    gen_var = np.var(x0_generated, axis=0)
    
    metrics['mean_error'] = float(np.mean(np.abs(gen_mean - target_mean)))
    metrics['var_error'] = float(np.mean(np.abs(gen_var - target_var)))
    
    # Higher moments
    target_skew = scipy.stats.skew(x0_target, axis=0)
    gen_skew = scipy.stats.skew(x0_generated, axis=0)
    metrics['skewness_error'] = float(np.mean(np.abs(gen_skew - target_skew)))
    
    target_kurt = scipy.stats.kurtosis(x0_target, axis=0)
    gen_kurt = scipy.stats.kurtosis(x0_generated, axis=0)
    metrics['kurtosis_error'] = float(np.mean(np.abs(gen_kurt - target_kurt)))
    
    # ====== DISTRIBUTIONAL METRICS ======
    
    # Maximum Mean Discrepancy (MMD) with RBF kernel
    try:
        metrics['mmd_rbf'] = float(compute_mmd_rbf(x0_target, x0_generated))
    except Exception as e:
        logger.debug(f"Could not compute MMD RBF: {e}")
        metrics['mmd_rbf'] = float('nan')
    
    # Energy distance (generalization of MMD)
    try:
        metrics['energy_distance'] = float(compute_energy_distance(x0_target, x0_generated))
    except Exception as e:
        logger.debug(f"Could not compute energy distance: {e}")
        metrics['energy_distance'] = float('nan')
    
    # Wasserstein distance (proper computation)
    try:
        metrics['wasserstein_distance'] = float(compute_wasserstein_distance(x0_target, x0_generated))
    except Exception as e:
        logger.debug(f"Could not compute Wasserstein distance: {e}")
        metrics['wasserstein_distance'] = float('nan')
    
    # Total Variation distance (for discrete distributions)
    try:
        metrics['total_variation'] = float(compute_total_variation_distance(x0_target, x0_generated))
    except Exception as e:
        logger.debug(f"Could not compute total variation distance: {e}")
        metrics['total_variation'] = float('nan')
    
    # ====== STATISTICAL TESTS ======
    
    # Two-sample Kolmogorov-Smirnov test (per dimension, then average p-value)
    try:
        ks_pvalues = []
        for dim in range(x0_target.shape[1]):
            _, p_val = scipy.stats.ks_2samp(x0_target[:, dim], x0_generated[:, dim])
            ks_pvalues.append(p_val)
        metrics['ks_test_pvalue'] = float(np.mean(ks_pvalues))
    except Exception as e:
        logger.debug(f"Could not compute KS test: {e}")
        metrics['ks_test_pvalue'] = float('nan')
    
    # Anderson-Darling test
    try:
        ad_statistics = []
        for dim in range(x0_target.shape[1]):
            # Combine samples and test if they come from same distribution
            combined = np.concatenate([x0_target[:, dim], x0_generated[:, dim]])
            labels = np.concatenate([np.zeros(len(x0_target)), np.ones(len(x0_generated))])
            
            # Use Mann-Whitney U test as a robust alternative
            _, p_val = scipy.stats.mannwhitneyu(x0_target[:, dim], x0_generated[:, dim], 
                                               alternative='two-sided')
            ad_statistics.append(p_val)
        metrics['mann_whitney_pvalue'] = float(np.mean(ad_statistics))
    except Exception as e:
        logger.debug(f"Could not compute Mann-Whitney test: {e}")
        metrics['mann_whitney_pvalue'] = float('nan')
    
    # ====== MULTIVARIATE METRICS ======
    
    # Frobenius norm of covariance difference
    try:
        target_cov = np.cov(x0_target.T)
        gen_cov = np.cov(x0_generated.T)
        metrics['covariance_frobenius'] = float(np.linalg.norm(target_cov - gen_cov, 'fro'))
    except Exception as e:
        logger.debug(f"Could not compute covariance Frobenius norm: {e}")
        metrics['covariance_frobenius'] = float('nan')
    
    # Sliced Wasserstein distance (efficient for high dimensions)
    try:
        metrics['sliced_wasserstein'] = float(compute_sliced_wasserstein(x0_target, x0_generated))
    except Exception as e:
        logger.debug(f"Could not compute sliced Wasserstein: {e}")
        metrics['sliced_wasserstein'] = float('nan')
    
    # ====== COUNT-SPECIFIC METRICS ======
    
    # For integer/count data, compute discrete-specific metrics
    if np.all(x0_target == x0_target.astype(int)) and np.all(x0_generated == x0_generated.astype(int)):
        # Chi-squared test for discrete distributions
        try:
            chi2_stats = []
            for dim in range(x0_target.shape[1]):
                # Get unique values and their counts
                all_values = np.union1d(x0_target[:, dim], x0_generated[:, dim])
                target_counts = np.array([np.sum(x0_target[:, dim] == v) for v in all_values])
                gen_counts = np.array([np.sum(x0_generated[:, dim] == v) for v in all_values])
                
                # Chi-squared test (avoid zeros)
                expected = (target_counts + gen_counts) / 2
                expected[expected == 0] = 1e-6
                
                chi2_stat = np.sum((target_counts - expected)**2 / expected) + \
                           np.sum((gen_counts - expected)**2 / expected)
                chi2_stats.append(chi2_stat)
            
            metrics['chi_squared_stat'] = float(np.mean(chi2_stats))
        except Exception as e:
            logger.debug(f"Could not compute chi-squared statistic: {e}")
            metrics['chi_squared_stat'] = float('nan')
    
    return metrics


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel."""
    from scipy.spatial.distance import cdist
    
    # Compute pairwise squared distances
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')
    
    # RBF kernel
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)
    
    # MMD² = E[K(X,X)] + E[K(Y,Y)] - 2E[K(X,Y)]
    mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return max(0, mmd_squared) ** 0.5  # Return MMD (not squared)


def compute_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute energy distance between two samples."""
    from scipy.spatial.distance import cdist
    
    # E[||X - Y||] - 0.5 * E[||X - X'||] - 0.5 * E[||Y - Y'||]
    XY_dist = np.mean(cdist(X, Y, 'sqeuclidean'))
    XX_dist = np.mean(cdist(X, X, 'sqeuclidean'))
    YY_dist = np.mean(cdist(Y, Y, 'sqeuclidean'))
    
    return 2 * XY_dist - XX_dist - YY_dist


def compute_wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Wasserstein distance using optimal transport."""
    try:
        import ot  # POT library for optimal transport
        
        # For multivariate case, approximate with sliced Wasserstein
        if X.shape[1] > 1:
            return compute_sliced_wasserstein(X, Y, n_projections=100)
        else:
            # 1D case: exact Wasserstein distance
            return ot.wasserstein_1d(X.flatten(), Y.flatten())
    
    except ImportError:
        # Fallback to 1D Wasserstein for each dimension
        wasserstein_dists = []
        for dim in range(X.shape[1]):
            x_sorted = np.sort(X[:, dim])
            y_sorted = np.sort(Y[:, dim])
            # Resample to same length if needed
            if len(x_sorted) != len(y_sorted):
                min_len = min(len(x_sorted), len(y_sorted))
                x_sorted = x_sorted[:min_len]
                y_sorted = y_sorted[:min_len]
            wasserstein_dists.append(np.mean((x_sorted - y_sorted)**2))
        return np.mean(wasserstein_dists)


def compute_sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_projections: int = 100) -> float:
    """Compute sliced Wasserstein distance."""
    d = X.shape[1]
    
    # Generate random projections
    projections = np.random.randn(d, n_projections)
    projections = projections / np.linalg.norm(projections, axis=0)
    
    wasserstein_distances = []
    
    for i in range(n_projections):
        # Project data onto random direction
        proj = projections[:, i]
        X_proj = X @ proj
        Y_proj = Y @ proj
        
        # 1D Wasserstein distance
        X_sorted = np.sort(X_proj)
        Y_sorted = np.sort(Y_proj)
        
        # Handle different lengths
        min_len = min(len(X_sorted), len(Y_sorted))
        X_sorted = X_sorted[:min_len]
        Y_sorted = Y_sorted[:min_len]
        
        wasserstein_distances.append(np.mean((X_sorted - Y_sorted)**2))
    
    return np.mean(wasserstein_distances)


def compute_total_variation_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute total variation distance for discrete distributions."""
    tv_distances = []
    
    for dim in range(X.shape[1]):
        # Get empirical distributions
        x_vals, x_counts = np.unique(X[:, dim], return_counts=True)
        y_vals, y_counts = np.unique(Y[:, dim], return_counts=True)
        
        # Normalize to probabilities
        x_probs = x_counts / len(X)
        y_probs = y_counts / len(Y)
        
        # Get union of support
        all_vals = np.union1d(x_vals, y_vals)
        
        # Create probability vectors
        x_prob_vec = np.zeros(len(all_vals))
        y_prob_vec = np.zeros(len(all_vals))
        
        for i, val in enumerate(all_vals):
            x_idx = np.where(x_vals == val)[0]
            y_idx = np.where(y_vals == val)[0]
            
            if len(x_idx) > 0:
                x_prob_vec[i] = x_probs[x_idx[0]]
            if len(y_idx) > 0:
                y_prob_vec[i] = y_probs[y_idx[0]]
        
        # Total variation distance = 0.5 * ||P - Q||_1
        tv_distance = 0.5 * np.sum(np.abs(x_prob_vec - y_prob_vec))
        tv_distances.append(tv_distance)
    
    return np.mean(tv_distances)


def aggregate_results(eval_files: List[Tuple[Path, Dict[str, Any]]]) -> pd.DataFrame:
    """Aggregate all results into a pandas DataFrame."""
    results = []
    all_param_keys = set()
    
    # First pass: collect all parameter keys
    for eval_path, config in eval_files:
        params = extract_experiment_params(config)
        all_param_keys.update(params.keys())
    
    # Second pass: ensure all experiments have all keys
    for eval_path, config in eval_files:
        # Extract experiment parameters
        params = extract_experiment_params(config)
        
        # Fill in missing parameters with 'None'
        for key in all_param_keys:
            if key not in params:
                params[key] = 'None'
        
        # Load evaluation data
        eval_data = load_evaluation_data(eval_path)
        if eval_data is None:
            continue
            
        # Compute metrics
        metrics = compute_comprehensive_metrics(eval_data)
        
        # Combine params and metrics
        result = {**params, **metrics}
        result['eval_path'] = str(eval_path)
        results.append(result)
    
    if not results:
        logger.error("No valid results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    logger.info(f"Aggregated {len(df)} experiment results")
    return df


def group_by_hyperparams(df: pd.DataFrame) -> pd.DataFrame:
    """Group results by hyperparameters (excluding seed) and compute statistics."""
    if df.empty:
        return df
    
    # Identify metric columns (computed by our analysis)
    metric_columns = [
        'mean_error', 'var_error', 'skewness_error', 'kurtosis_error',
        'mmd_rbf', 'energy_distance', 'wasserstein_distance', 'total_variation',
        'ks_test_pvalue', 'mann_whitney_pvalue', 'covariance_frobenius',
        'sliced_wasserstein', 'chi_squared_stat'
    ]
    
    # Exclude ONLY seed, eval_path, and computed metrics from grouping
    # Everything else (all config parameters) should be used for grouping
    exclude_columns = ['seed', 'eval_path'] + metric_columns
    grouping_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Group by hyperparameters
    grouped_results = []
    
    for group_keys, group_df in df.groupby(grouping_columns):
        if isinstance(group_keys, tuple):
            group_dict = dict(zip(grouping_columns, group_keys))
        else:
            group_dict = {grouping_columns[0]: group_keys}
        
        # Compute statistics across seeds
        result = group_dict.copy()
        result['n_seeds'] = len(group_df)
        result['seeds'] = sorted(group_df['seed'].tolist())
        
        for metric in metric_columns:
            if metric in group_df.columns:
                values = group_df[metric].dropna()
                if len(values) > 0:
                    result[f'{metric}_mean'] = float(np.mean(values))
                    result[f'{metric}_std'] = float(np.std(values))
                    result[f'{metric}_min'] = float(np.min(values))
                    result[f'{metric}_max'] = float(np.max(values))
                else:
                    result[f'{metric}_mean'] = float('nan')
                    result[f'{metric}_std'] = float('nan')
                    result[f'{metric}_min'] = float('nan')
                    result[f'{metric}_max'] = float('nan')
        
        grouped_results.append(result)
    
    grouped_df = pd.DataFrame(grouped_results)
    logger.info(f"Grouped into {len(grouped_df)} unique hyperparameter combinations")
    return grouped_df


def create_comparison_tables(grouped_df: pd.DataFrame, output_dir: str = "analysis_results"):
    """Create comparison tables for different aspects of the results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if grouped_df.empty:
        logger.warning("No grouped results to create tables from")
        return
    
    # Overall comparison table - use readable names
    display_columns = ['dataset_type', 'bridge_type', 'model_type', 'architecture_type', 'n_seeds',
                      'mean_error_mean', 'mean_error_std', 
                      'mmd_rbf_mean', 'mmd_rbf_std',
                      'wasserstein_distance_mean', 'wasserstein_distance_std',
                      'energy_distance_mean', 'energy_distance_std',
                      'total_variation_mean', 'total_variation_std']
    
    available_columns = [col for col in display_columns if col in grouped_df.columns]
    
    if available_columns:
        summary_table = grouped_df[available_columns].copy()
        
        # Round numerical columns
        numerical_cols = summary_table.select_dtypes(include=[np.number]).columns
        summary_table[numerical_cols] = summary_table[numerical_cols].round(4)
        
        # Save to CSV
        summary_path = output_path / "overall_comparison.csv"
        summary_table.to_csv(summary_path, index=False)
        logger.info(f"Saved overall comparison table to {summary_path}")
        
        # Print top performers (use MMD as primary metric, fallback to mean error)
        primary_metric = 'mmd_rbf_mean' if 'mmd_rbf_mean' in summary_table.columns else 'mean_error_mean'
        if primary_metric in summary_table.columns:
            print("\n" + "="*80)
            print(f"TOP 10 METHODS BY {primary_metric.upper().replace('_', ' ')}:")
            print("="*80)
            top_methods = summary_table.nsmallest(10, primary_metric)
            
            # Select display columns based on available metrics
            display_cols = ['dataset_type', 'bridge_type', 'model_type', 'architecture_type']
            if 'mmd_rbf_mean' in summary_table.columns:
                display_cols.extend(['mmd_rbf_mean', 'mmd_rbf_std'])
            if 'wasserstein_distance_mean' in summary_table.columns:
                display_cols.extend(['wasserstein_distance_mean'])
            if 'energy_distance_mean' in summary_table.columns:
                display_cols.extend(['energy_distance_mean'])
            
            available_display_cols = [col for col in display_cols if col in summary_table.columns]
            print(top_methods[available_display_cols].to_string(index=False))
    
    # Dataset-specific comparisons
    if 'dataset_type' in grouped_df.columns:
        for dataset in grouped_df['dataset_type'].unique():
            dataset_df = grouped_df[grouped_df['dataset_type'] == dataset]
            
            if len(dataset_df) > 1 and available_columns:
                dataset_table = dataset_df[available_columns].copy()
                numerical_cols = dataset_table.select_dtypes(include=[np.number]).columns
                dataset_table[numerical_cols] = dataset_table[numerical_cols].round(4)
                
                dataset_path = output_path / f"{dataset}_comparison.csv"
                dataset_table.to_csv(dataset_path, index=False)
                logger.info(f"Saved {dataset} comparison table to {dataset_path}")
    
    # Method comparison across datasets
    if 'bridge_type' in grouped_df.columns and 'dataset_type' in grouped_df.columns:
        method_comparison = grouped_df.groupby(['bridge_type', 'model_type']).agg({
            'mean_error_mean': ['mean', 'std', 'count'],
            'mmd_rbf_mean': ['mean', 'std', 'count'],
            'dataset_type': lambda x: list(x.unique())
        }).round(4)
        
        method_path = output_path / "method_comparison.csv"
        method_comparison.to_csv(method_path)
        logger.info(f"Saved method comparison table to {method_path}")


def create_visualizations(grouped_df: pd.DataFrame, output_dir: str = "analysis_results"):
    """Create visualizations of the results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if grouped_df.empty:
        logger.warning("No grouped results to create visualizations from")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Performance comparison plot
    plot_metric = 'mmd_rbf_mean' if 'mmd_rbf_mean' in grouped_df.columns else 'mean_error_mean'
    
    if all(col in grouped_df.columns for col in ['bridge_type', 'model_type', plot_metric]):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create method labels
        grouped_df['method'] = grouped_df['bridge_type'] + ' + ' + grouped_df['model_type']
        
        # Box plot or bar plot based on data
        if 'dataset_type' in grouped_df.columns and len(grouped_df['dataset_type'].unique()) > 1:
            sns.boxplot(data=grouped_df, x='method', y=plot_metric, hue='dataset_type', ax=ax)
            plt.xticks(rotation=45)
        else:
            sns.barplot(data=grouped_df, x='method', y=plot_metric, ax=ax)
            plt.xticks(rotation=45)
        
        metric_name = plot_metric.replace('_mean', '').replace('_', ' ').title()
        plt.title(f'{metric_name} Comparison Across Methods')
        plt.xlabel('Method (Bridge + Model)')
        plt.ylabel(metric_name)
        plt.tight_layout()
        
        plot_path = output_path / "method_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved method comparison plot to {plot_path}")
    
    # Learning rate analysis (if training.lr or similar exists)
    lr_cols = [col for col in grouped_df.columns if 'lr' in col.lower() or 'learning' in col.lower()]
    lr_col = lr_cols[0] if lr_cols else None
    
    if lr_col and plot_metric in grouped_df.columns and len(grouped_df[lr_col].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            # Convert to numeric for plotting
            grouped_df[f'{lr_col}_numeric'] = pd.to_numeric(grouped_df[lr_col], errors='coerce')
            plot_data = grouped_df.dropna(subset=[f'{lr_col}_numeric'])
            
            if 'bridge_type' in plot_data.columns:
                sns.scatterplot(data=plot_data, x=f'{lr_col}_numeric', y=plot_metric, hue='bridge_type', ax=ax)
            else:
                sns.scatterplot(data=plot_data, x=f'{lr_col}_numeric', y=plot_metric, ax=ax)
            
            plt.xscale('log')
            metric_name = plot_metric.replace('_mean', '').replace('_', ' ').title()
            plt.title(f'Learning Rate vs {metric_name}')
            plt.xlabel('Learning Rate')
            plt.ylabel(metric_name)
            plt.tight_layout()
            
            lr_plot_path = output_path / "learning_rate_analysis.png"
            plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved learning rate analysis plot to {lr_plot_path}")
            
        except Exception as e:
            logger.debug(f"Could not create learning rate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze counting flows experiment results")
    parser.add_argument("--base_dir", default="outputs", help="Base directory to search for results")
    parser.add_argument("--output_dir", default="analysis_results", help="Output directory for analysis")
    parser.add_argument("--multirun_dir", help="Specific multirun directory to analyze")
    
    args = parser.parse_args()
    
    # Determine search directory
    search_dir = args.multirun_dir if args.multirun_dir else args.base_dir
    
    print(f"Analyzing results from: {search_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Find all evaluation files
    eval_files = find_eval_files(search_dir)
    
    if not eval_files:
        print("No evaluation files found!")
        return
    
    # Aggregate results
    df = aggregate_results(eval_files)
    
    if df.empty:
        print("No valid results to analyze!")
        return
    
    # Group by hyperparameters
    grouped_df = group_by_hyperparams(df)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Save raw aggregated results
    df.to_csv(f"{args.output_dir}/raw_results.csv", index=False)
    grouped_df.to_csv(f"{args.output_dir}/grouped_results.csv", index=False)
    
    # Create comparison tables
    create_comparison_tables(grouped_df, args.output_dir)
    
    # Create visualizations
    create_visualizations(grouped_df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print(f"Raw results: {len(df)} experiments")
    print(f"Grouped results: {len(grouped_df)} unique hyperparameter combinations")


if __name__ == "__main__":
    main() 