#!/usr/bin/env python3
"""
Statistical testing script for neural dynamics quantile results.
This script loads saved quantile results and performs comprehensive statistical analyses
including mixed-effects models, post-hoc tests, and effect size calculations.

Usage:
    python dynamics_statistical_testing.py --analysis_type halfway --result_file quantile_results.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smt
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import mcnemar

# Import configuration
from avs_gazetime.config import PLOTS_DIR
from dynamics_params import EVENT_TYPE, CHANGE_METRIC, DELTA_T, ROI_GROUPS, HEMI
from dynamics_plot_params import DURATION_QUANTILES, ANALYSIS_TYPE
from dynamics_plotting import generate_filename_suffix

def load_quantile_results(result_file):
    """
    Load quantile results from CSV file.
    
    Parameters:
    -----------
    result_file : str
        Path to the CSV file containing quantile results.
        
    Returns:
    --------
    quantile_results : pd.DataFrame
        DataFrame with quantile results.
    """
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    quantile_results = pd.read_csv(result_file)
    print(f"Loaded {len(quantile_results)} observations from {result_file}")
    print(f"Columns: {list(quantile_results.columns)}")
    print(f"Subjects: {sorted(quantile_results['subject'].unique())}")
    print(f"ROIs: {sorted(quantile_results['roi'].unique())}")
    print(f"Quantiles: {sorted(quantile_results['duration_quantile'].unique())}")
    
    return quantile_results

def check_data_completeness(data, result_label):
    """
    Check data completeness and balance across conditions.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    summary : dict
        Dictionary with data completeness information.
    """
    print(f"\n=== Data Completeness Check ===")
    
    # Check for missing data
    missing_data = data[result_label].isnull().sum()
    total_data = len(data)
    print(f"Missing data: {missing_data}/{total_data} ({missing_data/total_data*100:.1f}%)")
    
    # Check balance across conditions
    condition_counts = data.groupby(['roi', 'duration_quantile']).size().reset_index(name='count')
    print(f"\nSample sizes per condition:")
    print(condition_counts.pivot(index='roi', columns='duration_quantile', values='count'))
    
    # Check subjects per condition
    subject_counts = data.groupby(['roi', 'duration_quantile'])['subject'].nunique().reset_index(name='n_subjects')
    print(f"\nNumber of subjects per condition:")
    print(subject_counts.pivot(index='roi', columns='duration_quantile', values='n_subjects'))
    
    # Identify incomplete cases
    complete_cases = data.groupby(['subject', 'roi']).size().reset_index(name='n_quantiles')
    expected_quantiles = len(DURATION_QUANTILES)
    incomplete_subjects = complete_cases[complete_cases['n_quantiles'] < expected_quantiles]
    
    if len(incomplete_subjects) > 0:
        print(f"\nIncomplete data for {len(incomplete_subjects)} subject-ROI combinations:")
        print(incomplete_subjects)
    
    return {
        'missing_data': missing_data,
        'total_data': total_data,
        'condition_counts': condition_counts,
        'subject_counts': subject_counts,
        'incomplete_subjects': incomplete_subjects
    }



def check_assumptions(data, result_label):
    """
    Check statistical assumptions for parametric tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    assumption_results : dict
        Dictionary with assumption test results.
    """
    print(f"\n=== Assumption Checking ===")
    
    data_clean = data.dropna(subset=[result_label])
    assumption_results = {}
    
    # 1. Normality tests per condition
    print("\n1. Normality Tests (Shapiro-Wilk):")
    normality_results = []
    
    for (roi, quantile), group in data_clean.groupby(['roi', 'duration_quantile']):
        if len(group) >= 3:  # Minimum for Shapiro-Wilk
            stat, p_value = stats.shapiro(group[result_label])
            normality_results.append({
                'roi': roi,
                'quantile': quantile,
                'n': len(group),
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            })
            print(f"  {roi} - {quantile}: W = {stat:.4f}, p = {p_value:.4f}")
    
    normality_df = pd.DataFrame(normality_results)
    assumption_results['normality'] = normality_df
    
    # 2. Homogeneity of variance (Levene's test) per ROI
    print("\n2. Homogeneity of Variance (Levene's Test):")
    variance_results = []
    
    for roi in data_clean['roi'].unique():
        roi_data = data_clean[data_clean['roi'] == roi]
        groups = [group[result_label].values for _, group in roi_data.groupby('duration_quantile')]
        
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            stat, p_value = stats.levene(*groups)
            variance_results.append({
                'roi': roi,
                'statistic': stat,
                'p_value': p_value,
                'homogeneous': p_value > 0.05
            })
            print(f"  {roi}: W = {stat:.4f}, p = {p_value:.4f}")
    
    variance_df = pd.DataFrame(variance_results)
    assumption_results['variance_homogeneity'] = variance_df
    
    # 3. Sphericity (for repeated measures) - approximated by compound symmetry
    print("\n3. Sphericity Assessment:")
    print("  Note: Exact sphericity test requires raw data. Mixed-effects models are robust to sphericity violations.")
    
    return assumption_results

def mixed_effects_analysis(data, result_label):
    """
    Perform mixed-effects linear model analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    model_results : dict
        Dictionary with model results.
    """
    print(f"\n=== Mixed-Effects Linear Model ===")
    
    # Clean data
    data_clean = data.dropna(subset=[result_label])
    print(f"Analysis with {len(data_clean)} complete observations")
    
    # Check if hemisphere column exists
    has_hemisphere = 'hemisphere' in data_clean.columns
    print(data_clean.columns)
    try:
        # Fit mixed-effects model
        if has_hemisphere:
            formula = f"{result_label} ~ C(duration_quantile, Treatment(reference='very short')) + C(roi) + C(hemisphere)"
        else:
            formula = f"{result_label} ~ C(duration_quantile, Treatment(reference='very short')) + C(roi)"
        data_clean['subject_roi'] = data['subject'].astype(str) + '_' + data['roi'].astype(str)

        print(f"Model formula: {formula}")
        
        model = smf.mixedlm(formula, data=data_clean, groups=data_clean["subject"])#, re_formula="1")
        result = model.fit()
        
        print(f"\nModel Summary:")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")
        print(f"Log-Likelihood: {result.llf:.2f}")
        
        # Extract fixed effects
        print(f"\nFixed Effects:")
        fixed_effects = []
        for param in result.params.index:
            coef = result.params[param]
            se = result.bse[param]
            t_val = result.tvalues[param]
            p_val = result.pvalues[param]
            ci_lower = result.conf_int().loc[param, 0]
            ci_upper = result.conf_int().loc[param, 1]
            
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {param}: β = {coef:.4f} ({se:.4f}), t = {t_val:.2f}, p = {p_val:.4f} {significance}")
            print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            fixed_effects.append({
                'parameter': param,
                'coefficient': coef,
                'se': se,
                't_value': t_val,
                'p_value': p_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })
        
        # Extract random effects
        print(f"\nRandom Effects Variance:")
        print(f"  Subject (Intercept): {result.cov_re.iloc[0, 0]:.6f}")
        print(f"  Residual: {result.scale:.6f}")
        
        return {
            'model': result,
            'fixed_effects': pd.DataFrame(fixed_effects),
            'formula': formula,
            'aic': result.aic,
            'bic': result.bic,
            'loglik': result.llf
        }
        
    except Exception as e:
        print(f"Mixed-effects model failed: {e}")
        return None

def post_hoc_analysis(data, result_label):
    """
    Perform post-hoc pairwise comparisons.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    posthoc_results : dict
        Dictionary with post-hoc results.
    """
    print(f"\n=== Post-Hoc Analysis ===")
    
    data_clean = data.dropna(subset=[result_label])
    posthoc_results = {}
    
    # Pairwise comparisons for each ROI
    print("\nPairwise Comparisons (Wilcoxon Signed-Rank Tests):")
    
    from itertools import combinations
    
    for roi in data_clean['roi'].unique():
        print(f"\n{roi}:")
        roi_data = data_clean[data_clean['roi'] == roi]
        
        # Get data for each quantile
        quantile_data = {}
        for quantile in DURATION_QUANTILES:
            q_data = roi_data[roi_data['duration_quantile'] == quantile]
            if len(q_data) > 0:
                quantile_data[quantile] = q_data[result_label].values
        
        # Pairwise comparisons
        comparisons = []
        p_values = []
        
        for q1, q2 in combinations(quantile_data.keys(), 2):
            if len(quantile_data[q1]) > 0 and len(quantile_data[q2]) > 0:
                # Use paired test if same subjects, otherwise unpaired
                if len(quantile_data[q1]) == len(quantile_data[q2]):
                    stat, p_value = stats.wilcoxon(quantile_data[q1], quantile_data[q2])
                    test_type = "Wilcoxon signed-rank"
                else:
                    stat, p_value = stats.mannwhitneyu(quantile_data[q1], quantile_data[q2])
                    test_type = "Mann-Whitney U"
                
                comparisons.append(f"{q1} vs {q2}")
                p_values.append(p_value)
                
                print(f"  {q1} vs {q2}: {test_type}, p = {p_value:.4f}")
        
        # Multiple comparison correction
        if len(p_values) > 0:
            rejected, p_corrected, _, _ = smt.multipletests(p_values, method='fdr_bh')
            print(f"fdr_bh-corrected p-values:")
            for comp, p_orig, p_corr, reject in zip(comparisons, p_values, p_corrected, rejected):
                significance = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""
                print(f"    {comp}: p = {p_corr:.4f} {significance}")
            
        
        posthoc_results[roi] = {
            'comparisons': comparisons,
            'p_values': p_values,
            'p_corrected': p_corrected if len(p_values) > 0 else [],
            'rejected': rejected if len(p_values) > 0 else []
        }
    
    return posthoc_results

def effect_size_analysis(data, result_label):
    """
    Calculate effect sizes for quantile differences.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    effect_sizes : pd.DataFrame
        DataFrame with effect sizes.
    """
    print(f"\n=== Effect Size Analysis ===")
    
    data_clean = data.dropna(subset=[result_label])
    effect_sizes = []
    
    from itertools import combinations
    
    def cohens_d(x, y):
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    print("Cohen's d effect sizes (small: 0.2, medium: 0.5, large: 0.8):")
    
    for roi in data_clean['roi'].unique():
        print(f"\n{roi}:")
        roi_data = data_clean[data_clean['roi'] == roi]
        
        # Get data for each quantile
        quantile_data = {}
        for quantile in DURATION_QUANTILES:
            q_data = roi_data[roi_data['duration_quantile'] == quantile]
            if len(q_data) > 0:
                quantile_data[quantile] = q_data[result_label].values
        
        # Calculate effect sizes for pairwise comparisons
        for q1, q2 in combinations(quantile_data.keys(), 2):
            if len(quantile_data[q1]) > 0 and len(quantile_data[q2]) > 0:
                d = cohens_d(quantile_data[q1], quantile_data[q2])
                magnitude = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small" if abs(d) >= 0.2 else "negligible"
                
                effect_sizes.append({
                    'roi': roi,
                    'comparison': f"{q1} vs {q2}",
                    'cohens_d': d,
                    'magnitude': magnitude,
                    'q1_mean': np.mean(quantile_data[q1]),
                    'q2_mean': np.mean(quantile_data[q2]),
                    'q1_n': len(quantile_data[q1]),
                    'q2_n': len(quantile_data[q2])
                })
                
                print(f"  {q1} vs {q2}: d = {d:.3f} ({magnitude})")
    
    return pd.DataFrame(effect_sizes)

def nonparametric_analysis(data, result_label):
    """
    Perform nonparametric analyses as alternatives to parametric tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable.
        
    Returns:
    --------
    nonparam_results : dict
        Dictionary with nonparametric test results.
    """
    print(f"\n=== Nonparametric Analysis ===")
    
    data_clean = data.dropna(subset=[result_label])
    nonparam_results = {}
    
    # Friedman test for each ROI (repeated measures alternative)
    print("Friedman Test (nonparametric repeated measures ANOVA):")
    
    friedman_results = []
    for roi in data_clean['roi'].unique():
        roi_data = data_clean[data_clean['roi'] == roi]
        
        # Prepare data matrix (subjects x quantiles)
        pivot_data = roi_data.pivot_table(
            index='subject', 
            columns='duration_quantile', 
            values=result_label
        )
        
        # Remove subjects with missing data
        pivot_data = pivot_data.dropna()
        
        if len(pivot_data) >= 3 and len(pivot_data.columns) >= 3:
            # Perform Friedman test
            stat, p_value = friedmanchisquare(*[pivot_data[col].values for col in pivot_data.columns])
            
            friedman_results.append({
                'roi': roi,
                'n_subjects': len(pivot_data),
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {roi}: χ² = {stat:.4f}, p = {p_value:.4f} {significance} (n = {len(pivot_data)})")
        else:
            print(f"  {roi}: Insufficient data for Friedman test")
    
    friedman_df = pd.DataFrame(friedman_results)
    nonparam_results['friedman'] = friedman_df
    
    # Multiple comparison correction for Friedman tests
    if len(friedman_results) > 0:
        p_values = [r['p_value'] for r in friedman_results]
        rejected, p_corrected, _, _ = smt.multipletests(p_values, method='fdr_bh')
        
        print(f"\Benjamini-Hochberg corrected p-values for Friedman's test:")
        for i, roi_result in enumerate(friedman_results):
            significance = "***" if p_corrected[i] < 0.001 else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else ""
            print(f"  {roi_result['roi']}: corrected p = {p_corrected[i]:.4f} {significance}")
    else:
        print("No Friedman's test results to correct.")
    return nonparam_results

def save_results(output_dir, **results):
    """
    Save all statistical results to files.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save results.
    **results : dict
        Various analysis results to save.
    """
    print(f"\n=== Saving Results ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save descriptives
    if 'descriptives' in results:
        desc_file = os.path.join(output_dir, "descriptive_statistics.csv")
        results['descriptives'].to_csv(desc_file, index=False)
        print(f"Saved descriptive statistics to {desc_file}")
    
    # Save mixed-effects results
    if 'mixed_effects' in results and results['mixed_effects'] is not None:
        me_file = os.path.join(output_dir, "mixed_effects_results.csv")
        results['mixed_effects']['fixed_effects'].to_csv(me_file, index=False)
        print(f"Saved mixed-effects results to {me_file}")
    
    # Save effect sizes
    if 'effect_sizes' in results:
        es_file = os.path.join(output_dir, "effect_sizes.csv")
        results['effect_sizes'].to_csv(es_file, index=False)
        print(f"Saved effect sizes to {es_file}")
    
    # Save Friedman results
    if 'nonparametric' in results and 'friedman' in results['nonparametric']:
        friedman_file = os.path.join(output_dir, "friedman_results.csv")
        results['nonparametric']['friedman'].to_csv(friedman_file, index=False)
        print(f"Saved Friedman test results to {friedman_file}")
    
    return


def plot_catplot_results(data, result_label, output_dir, plot_type='strip', hue_order=None, figsize=(14, 10)):
    """
    Create a categorical plot of neural dynamics results across ROIs and duration quantiles.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with quantile results.
    result_label : str
        Name of the dependent variable ('t_halfway', 'peak_latency', or 'peak_amplitude').
    output_dir : str
        Directory to save the plots.
    plot_type : str, optional
        Type of catplot to create ('boxen', 'box', 'violin', 'point', 'bar', or 'strip').
    hue_order : list, optional
        Order of duration quantiles for consistent coloring. If None, uses alphabetical order.
    figsize : tuple, optional
        Figure size (width, height).
        
    Returns:
    --------
    g : seaborn.FacetGrid
        The resulting seaborn FacetGrid object for further customization if needed.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    print(f"\n=== Creating {plot_type.capitalize()} Plot of Results ===")
    
    # Clean data and convert to milliseconds if time-based
    data_clean = data.dropna(subset=[result_label])
    
    if result_label in ['t_halfway', 'peak_latency']:
        data_clean = data_clean.copy()
        data_clean[result_label] = data_clean[result_label] * 1000  # Convert to ms
    
    # Set seaborn context
    sns.set_context("poster")
    sns.set_style("ticks")
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Get ROIs and set hue order if not provided
    roi_names = sorted(data_clean['roi'].unique())
    if hue_order is None:
        hue_order = sorted(data_clean['duration_quantile'].unique())
        # inverce
        hue_order = hue_order[::-1]
    
    # Set color palette
    palette = "magma"
    
    # Create the catplot
    g = sns.catplot(
        data=data_clean,
        x="roi",
        y=result_label,
        hue="duration_quantile",
        kind=plot_type,
        palette=palette,
        hue_order=hue_order,
        dodge=True,
        height=figsize[1]/3,
        aspect=figsize[0]/figsize[1]*3,
        legend_out=True,
        errorbar=('ci', 95),
        #scale="width"
    )
    
    # Customize plot appearance
    g.despine()
    
    # Adjust title and axis labels
    if result_label == 't_halfway':
        y_label = "Pattern change\npost-peak halfway latency [ms]"
        title = "Pattern Change Halfway Latency by ROI and Fixation Duration"
    elif result_label == 'peak_latency':
        y_label = "Peak latency [ms]"
        title = "Peak Latency by ROI and Fixation Duration"
    elif result_label == 'peak_amplitude':
        y_label = "Peak amplitude [normalized]"
        title = "Peak Amplitude by ROI and Fixation Duration"
    else:
        y_label = result_label
        title = f"{result_label} by ROI and Fixation Duration"
    
    g.set_axis_labels("ROI", y_label)
    g.fig.suptitle(title, y=1.02, fontsize=20)
    
    # Add reference line for zero if relevant
    if result_label in ['t_halfway', 'peak_latency']:
        # Add reference line at 0 ms
        for ax in g.axes.flat:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Adjust legend
    g._legend.set_title("fixation duration")
    
    # Add significance markers from statistical tests
    # (This would require the statistical results as input, for example:)
    # for roi in roi_names:
    #    # Find the axis for this ROI
    #    ax_idx = roi_names.index(roi)
    #    ax = g.axes.flat[ax_idx]
    #    # Add significance markers based on statistical tests
    #    # ...
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(roi_names) > 6 else 0)
    
    # Adjust layout
   # plt.tight_layout()
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{result_label}_{plot_type}_plot"
    fig_path = os.path.join(output_dir, filename)
    g.fig.savefig(f"{fig_path}.png", dpi=300, bbox_inches='tight')
    g.fig.savefig(f"{fig_path}.pdf", bbox_inches='tight')
    
    print(f"Saved {plot_type} plot to {fig_path}.png and {fig_path}.pdf")
    
    return g

def main():
    """Main function to run statistical analysis."""
    
    output_dir = os.path.join(PLOTS_DIR, "dynamics", "analysis")
    suffix = generate_filename_suffix()
    result_label = "t_halfway"
    # /share/klab/psulewski/psulewski/active-visual-semantics-MEG/results/fullrun/analysis/gazetime/submission_checks/ica/stc/filter_0.2_200/as01/dynamics/analysis/quantile_results_t_halfway_fixation_correlation_both_30ms.csv
    filename = f"quantile_results_{result_label}_{EVENT_TYPE}_{CHANGE_METRIC}_{HEMI}_{DELTA_T}ms{suffix}.csv"
    
    filename_results = os.path.join(PLOTS_DIR, "dynamics", "analysis", filename)
    data = load_quantile_results(filename_results)
    # Run analyses
    results = {}
    
    # 1. Data completeness check
    results['data_check'] = check_data_completeness(data, result_label)

    # 3. Assumption checking
    #results['assumptions'] = check_assumptions(data, result_label)
    
    # 4. Mixed-effects analysis
    results['mixed_effects'] = mixed_effects_analysis(data, result_label)
    
    # 5. Post-hoc analysis
    results['posthoc'] = post_hoc_analysis(data, result_label)
    
    # 6. Effect size analysis
    #results['effect_sizes'] = effect_size_analysis(data, result_label)
    
    # 7. Nonparametric analysis
    #results['nonparametric'] = nonparametric_analysis(data, result_label)

    
    plot_catplot_results(data, result_label, output_dir, plot_type='strip', hue_order=None, figsize=(10, 15))
    

if __name__ == "__main__":
    main()