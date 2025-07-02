"""
Clean script for fix2cap data quality checks
Focuses on fixation duration analysis and example image generation
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from avs_gazetime.utils.tools import compute_quantiles, get_quantile_data

# Configuration
rater_ids = ["ld", 'og_all']
fix2cap_dir = '/share/klab/psulewski/psulewski/active-visual-semantics-MEG/results/fullrun/analysis/gazetime/fix2cap'
log_duration = False
subjects = [1, 2, 3, 4, 5]
plot_examples_images = 10  # Number of examples or False
compute_cohens_kappa = True
plt.rcParams.update({'font.size': 15})

# Load and prepare data
print("Loading data...")
for r, rater_id in enumerate(rater_ids):
    fix2cap_file = os.path.join(fix2cap_dir, "et", f'fix2cap_events_{rater_id}.csv')
    fix2cap_rater = pd.read_csv(fix2cap_file)
    fix2cap_rater['rater_id'] = rater_id
    if r == 0:
        fix2cap = fix2cap_rater
    else:
        fix2cap = pd.concat([fix2cap, fix2cap_rater])

# Data preprocessing
print(f"Proportion of data done: {fix2cap['fix2cap_done'].sum() / len(fix2cap):.3f}")
fix2cap_done = fix2cap[fix2cap['fix2cap_done'] == True].copy()
fix2cap_done = fix2cap_done.drop(columns=[col for col in fix2cap_done.columns if 'Unamed' in col])

# Create output directory
rater_id_combined = "all" if len(rater_ids) > 1 else rater_ids[0]
fix2cap_dir_out = os.path.join(fix2cap_dir, "quality_checks", rater_id_combined)
os.makedirs(fix2cap_dir_out, exist_ok=True)

# Duration preprocessing
if log_duration:
    fix2cap_done['duration'] = np.log(fix2cap_done['duration'])
    y_label = "log fixation duration [ms]"
else:
    y_label = "fixation duration [ms]"

fix2cap_done[fix2cap_done.fix_sequence_from_last != -1]

# Remove outliers per subject using groupby
def filter_outliers_by_subject(group, lower_pct=0.02, upper_pct=0.98):
    """Filter outliers within each subject group."""
    duration_low = group['duration'].quantile(lower_pct)
    duration_high = group['duration'].quantile(upper_pct)
    return group[(group['duration'] > duration_low) & (group['duration'] < duration_high)]

# Apply to your data (replace 'subject' with your actual subject column name)
fix2cap_done = fix2cap_done.groupby('subject').apply(filter_outliers_by_subject).reset_index(drop=True)


# Compute quantiles and create condition categories
n_quantiles = 80
fix2cap_done = compute_quantiles(fix2cap_done, dur_col='duration', quantiles=n_quantiles)

# Create none_style column with correct ordering
fix2cap_done['none_style'] = 'false'  # default
fix2cap_done.loc[fix2cap_done['none_from_other_subject'] != "0.0", 'none_style'] = 'other'
fix2cap_done.loc[fix2cap_done['in_caption'] == True, 'none_style'] = 'self'

# Convert to categorical with desired order
fix2cap_done['none_style'] = pd.Categorical(
    fix2cap_done['none_style'], 
    categories=['self', 'false', 'other'], 
    ordered=True
)

# Convert duration to milliseconds
fix2cap_done['duration'] = fix2cap_done['duration'] * 1000

print(f"Condition counts:\n{fix2cap_done['none_style'].value_counts()}")

# STATISTICAL ANALYSIS
print("\n=== STATISTICAL ANALYSIS ===")

# 1. Descriptive statistics
desc_stats = fix2cap_done.groupby('none_style')['duration'].agg(['count', 'mean', 'std', 'median'])
print(f"\nDescriptive statistics:\n{desc_stats}")

# 2. Test for normality
print(f"\nShapiro-Wilk test for normality (sample of 5000):")
for condition in ['self', 'false', 'other']:
    sample = fix2cap_done[fix2cap_done['none_style'] == condition]['duration'].sample(min(5000, len(fix2cap_done[fix2cap_done['none_style'] == condition])))
    stat, p = stats.shapiro(sample)
    print(f"{condition}: W={stat:.4f}, p={p:.2e}")

# 3. Kruskal-Wallis test (non-parametric)
groups = [fix2cap_done[fix2cap_done['none_style'] == condition]['duration'] for condition in ['self', 'false', 'other']]
h_stat, kw_p = stats.kruskal(*groups)
print(f"\nKruskal-Wallis test: H={h_stat:.4f}, p={kw_p:.2e}")

# 4. Post-hoc pairwise comparisons (Mann-Whitney U with Bonferroni correction)
from scipy.stats import mannwhitneyu
pairs = [('self', 'false'), ('self', 'other'), ('false', 'other')]
print(f"\nPost-hoc pairwise comparisons (Mann-Whitney U):")
for pair in pairs:
    group1 = fix2cap_done[fix2cap_done['none_style'] == pair[0]]['duration']
    group2 = fix2cap_done[fix2cap_done['none_style'] == pair[1]]['duration']
    u_stat, p = mannwhitneyu(group1, group2)
    p_bonf = p * 3  # Bonferroni correction
    print(f"{pair[0]} vs {pair[1]}: U={u_stat:.0f}, p={p:.2e}, p_bonf={p_bonf:.2e}")

# 5. Mixed-effects model
print(f"\nMixed-effects model:")
try:
    model = smf.mixedlm("duration ~ none_style", fix2cap_done, 
                       groups=fix2cap_done["subject"],
                      )
    result = model.fit()
    print(result.summary())
    
    # Test false vs other directly by changing reference category
    fix2cap_done_reref = fix2cap_done.copy()
    fix2cap_done_reref['none_style'] = pd.Categorical(
        fix2cap_done_reref['none_style'], 
        categories=['false', 'self', 'other'], 
        ordered=True
    )
    
    model_reref = smf.mixedlm("duration ~ none_style", fix2cap_done_reref, 
                             groups=fix2cap_done_reref["subject"],
                            )
    result_reref = model_reref.fit()
    print(result_reref.summary())
    # Extract false vs other comparison
    other_coef = result_reref.params['none_style[T.other]']
    other_ci = result_reref.conf_int().loc['none_style[T.other]']
    other_p = result_reref.pvalues['none_style[T.other]']
    
    print(f"\nFalse vs Other comparison:")
    print(f"β = {other_coef:.1f} ms, 95% CI [{other_ci[0]:.1f}, {other_ci[1]:.1f}], p = {other_p:.3f}")
    
except Exception as e:
    print(f"Mixed-effects model failed: {e}")

# MAIN PLOTTING
print("\n=== GENERATING PLOTS ===")
sns.set_palette("colorblind")
sns.set_context("poster")

# Main duration comparison plot
fig, [ax, ax_point] = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Box plot
sns.boxplot(x="none_style", y="duration", data=fix2cap_done, ax=ax, 
           hue='none_style', legend=False, showfliers=False)

# Point plot with confidence intervals
sns.pointplot(x="none_style", y="duration", data=fix2cap_done, ax=ax_point, 
             hue='none_style', ci=95, dodge=True, estimator=np.mean, legend=False)

# Add individual rater lines (very faint grey)
if len(rater_ids) > 1:
    for rater in rater_ids:
        rater_data = fix2cap_done[fix2cap_done['rater_id'] == rater]
        rater_means = rater_data.groupby('none_style')['duration'].mean()
        ax_point.plot(range(len(rater_means)), rater_means.values, 
                     color='grey', alpha=0.3, linewidth=2, linestyle='--')

sns.despine()
ax.set_ylabel(y_label)
ax.set_xlabel("used in scene caption")
ax_point.set_ylabel(f"mean {y_label}")
ax_point.set_xlabel("used in scene caption")

fig.tight_layout()
fig.savefig(os.path.join(fix2cap_dir_out, f'fix2cap_duration_comparison_{rater_id_combined}_log_{log_duration}.pdf'))
plt.show()

# Quantile distribution plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=fix2cap_done, x='quantile', hue='none_style', multiple='fill',
             ax=ax, stat='count', element="step", bins=n_quantiles, 
             hue_order=['self', 'false', 'other'], alpha=0.7)

ax.set_xlabel("Fixation duration quantile")
ax.set_ylabel("Proportion of fixations")
fig.tight_layout()
fig.savefig(os.path.join(fix2cap_dir_out, f'fix2cap_quantile_distribution_{rater_id_combined}.pdf'))
plt.show()

# Subject-wise analysis
fig, ax = plt.subplots(figsize=(8, 6))
fix_done_per_subject = fix2cap_done.groupby('subject').size()
fix2cap_done_sub = fix2cap_done[fix2cap_done['subject'].isin(
    fix_done_per_subject[fix_done_per_subject > 1000].index
)]

sns.pointplot(x="none_style", y="duration", data=fix2cap_done_sub, ax=ax, 
             hue='subject', ci=95, dodge=False, estimator=np.mean, legend=True)
ax.set_ylabel(f"Mean {y_label}")
ax.set_xlabel("Fixation target category")
fig.tight_layout()
fig.savefig(os.path.join(fix2cap_dir_out, f'fix2cap_by_subject_{rater_id_combined}.pdf'))
plt.show()

# EXAMPLE IMAGES
if plot_examples_images:
    print(f"\n=== GENERATING {plot_examples_images} EXAMPLE IMAGES ===")
    try:
        from avs_gazetime.fix2cap.fixation2caption_matcher import Fixation2CaptionMatcher
        fix2cap_matcher = Fixation2CaptionMatcher(fix2cap_dir)
        
        unique_scenes = fix2cap_done['sceneID'].unique()
        np.random.seed(42)
        selected_scenes = np.random.choice(unique_scenes, min(plot_examples_images, len(unique_scenes)), replace=False)
        
        for sceneID in selected_scenes:
            events_scene = fix2cap_done[fix2cap_done['sceneID'] == sceneID]
            
            # Get scene image
            im_rescaled, im_width_rescaled, im_height_rescaled = fix2cap_matcher.get_scene_image(sceneID)
            
            # Prepare fixation data
            x = events_scene['mean_gx'].values
            y = events_scene['mean_gy'].values
            words = events_scene['fix_word'].values.copy()
            words[words == "0.0"] = "None"
            
            # Color code by condition
            word_colors = []
            for _, event in events_scene.iterrows():
                if event['none_style'] == 'self':
                    word_colors.append('white')
                elif event['none_style'] == 'other':
                    word_colors.append('cyan')
                    # Use label from other subject
                    idx = events_scene.index.get_loc(event.name)
                    words[idx] = event['none_from_other_subject']
                else:  # false
                    word_colors.append('magenta')
                    if event['none_typed'] != "0.0" and str(event['none_typed']) != "nan":
                        idx = events_scene.index.get_loc(event.name)
                        words[idx] = event['none_typed']
            
            # Create plot
            plot_fname = fix2cap_matcher.mark_fixation_target(
                x, y, im_rescaled, im_width_rescaled, im_height_rescaled, 
                words=words, word_color=word_colors
            )
            
            plot = plt.imread(plot_fname)
            plt.figure(figsize=(12, 8))
            plt.imshow(plot)
            plt.axis('off')
            
            caption = events_scene['transcribed_caption'].iloc[0]
            plt.title(f"Scene {sceneID}\n{caption}", fontsize=12, wrap=True)
            plt.tight_layout()
            plt.savefig(os.path.join(fix2cap_dir_out, f'example_scene_{sceneID}_{rater_id_combined}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.show()
            
    except ImportError as e:
        print(f"Could not import Fixation2CaptionMatcher: {e}")
    except Exception as e:
        print(f"Error generating example images: {e}")

print(f"\nAnalysis complete. Output saved to: {fix2cap_dir_out}")

def compute_cohens_kappa_fix2cap(fix2cap_data, rater_col='rater_id', 
                                 response_col='none_style', fix2cap_dir_out=None):
    """
    Compute Cohen's kappa for inter-rater reliability and plot confusion matrices.
    Computes both 3-category (self/false/other) and binary (self vs not-self) kappa.
    
    Parameters:
    -----------
    fix2cap_data : pd.DataFrame
        DataFrame containing fix2cap ratings from multiple raters
    rater_col : str
        Column name for rater identifiers
    response_col : str  
        Column name for categorical responses ('self', 'false', 'other')
    fix2cap_dir_out : str
        Output directory for saving plots
        
    Returns:
    --------
    results : dict
        Dictionary with 'three_way' and 'binary' kappa values
    """
    from sklearn.metrics import cohen_kappa_score
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Get unique raters
    raters = fix2cap_data[rater_col].unique()
    
    if len(raters) != 2:
        raise ValueError("This function requires exactly 2 raters")
    
    rater1, rater2 = raters
    
    # Get overlapping rated items
    r1_data = fix2cap_data[fix2cap_data[rater_col] == rater1]
    r2_data = fix2cap_data[fix2cap_data[rater_col] == rater2]
    
    # Merge on identifying columns
    id_cols = ['subject', 'trial', 'fix_sequence', 'sceneID']
    id_cols = [col for col in id_cols if col in fix2cap_data.columns]
    
    merged = pd.merge(r1_data, r2_data, on=id_cols, suffixes=('_r1', '_r2'))
    
    if len(merged) == 0:
        raise ValueError(f"No overlapping ratings between {rater1} and {rater2}")
    
    # Extract ratings
    ratings_r1 = merged[f"{response_col}_r1"]
    ratings_r2 = merged[f"{response_col}_r2"]
    
    # Set context
    sns.set_context("poster")
    
    # 1. Three-way kappa (self, false, other)
    kappa_3way = cohen_kappa_score(ratings_r1, ratings_r2)
    agreement_3way = (ratings_r1 == ratings_r2).mean()
    
    confusion_3way = pd.crosstab(ratings_r1, ratings_r2, 
                                rownames=[rater1], colnames=[rater2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 3-way confusion matrix
    sns.heatmap(confusion_3way, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'3-way Agreement\nκ = {kappa_3way:.3f}, Agreement = {agreement_3way:.1%}')
    ax1.set_ylabel(f'{rater1} ratings')
    ax1.set_xlabel(f'{rater2} ratings')
    
    # 2. Binary kappa (self vs not-self)
    ratings_r1_binary = (ratings_r1 == 'self').astype(str)
    ratings_r2_binary = (ratings_r2 == 'self').astype(str)
    ratings_r1_binary = ratings_r1_binary.replace({'True': 'self', 'False': 'not-self'})
    ratings_r2_binary = ratings_r2_binary.replace({'True': 'self', 'False': 'not-self'})
    
    kappa_binary = cohen_kappa_score(ratings_r1_binary, ratings_r2_binary)
    agreement_binary = (ratings_r1_binary == ratings_r2_binary).mean()
    
    confusion_binary = pd.crosstab(ratings_r1_binary, ratings_r2_binary,
                                  rownames=[rater1], colnames=[rater2])
    
    # Plot binary confusion matrix
    sns.heatmap(confusion_binary, annot=True, fmt='d', cmap='Oranges', ax=ax2)
    ax2.set_title(f'Binary Agreement\nκ = {kappa_binary:.3f}, Agreement = {agreement_binary:.1%}')
    ax2.set_ylabel(f'{rater1} ratings')
    ax2.set_xlabel(f'{rater2} ratings')
    
    plt.tight_layout()
    
    if fix2cap_dir_out:
        plt.savefig(os.path.join(fix2cap_dir_out, f'cohens_kappa_{rater1}_{rater2}.pdf'),
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print results
    print(f"=== THREE-WAY KAPPA (self/false/other) ===")
    print(f"Cohen's kappa: {kappa_3way:.3f}")
    print(f"Percent agreement: {agreement_3way:.1%}")
    print(f"Overlapping ratings: {len(merged)}")
    
    print(f"\n=== BINARY KAPPA (self vs not-self) ===")
    print(f"Cohen's kappa: {kappa_binary:.3f}")
    print(f"Percent agreement: {agreement_binary:.1%}")
    
    # Interpretation function
    def interpret_kappa(kappa):
        if kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost perfect"
    
    print(f"3-way interpretation: {interpret_kappa(kappa_3way)}")
    print(f"Binary interpretation: {interpret_kappa(kappa_binary)}")
    
    return {
        'three_way': kappa_3way,
        'binary': kappa_binary,
        'agreement_3way': agreement_3way,
        'agreement_binary': agreement_binary,
        'n_overlapping': len(merged)
    }


if compute_cohens_kappa:
    print("\n=== COMPUTING COHEN'S KAPPA ===")
   
    kappa = compute_cohens_kappa_fix2cap(fix2cap_done, 
                                              rater_col='rater_id', 
                                              response_col='none_style', 
                                              fix2cap_dir_out=fix2cap_dir_out)
   
    
    

def bootstrap_ci(data, n_bootstrap=5000, ci=0.95):
    """
    Calculate bootstrapped confidence intervals for a given dataset.
    
    Parameters:
    -----------
    data : array-like
        Input data for bootstrapping
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 5000)
    ci : float, optional
        Confidence interval width (default: 0.95 for 95% CI)
    
    Returns:
    --------
    dict
        Dictionary containing mean, lower and upper confidence interval bounds
    """
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    
    # Calculate confidence intervals
    lower_percentile = (1 - ci) / 2
    upper_percentile = 1 - lower_percentile
    
    return {
        'mean': np.mean(data),
        'ci_lower': np.percentile(bootstrap_means, lower_percentile * 100),
        'ci_upper': np.percentile(bootstrap_means, upper_percentile * 100)
    }


mean_durations = {}
ci_durations = {}

for category in fix2cap_done['none_style'].unique():
    category_data = fix2cap_done[fix2cap_done['none_style'] == category]['duration']
    
    bootstrap_result = bootstrap_ci(category_data)
    
    mean_durations[category] = bootstrap_result['mean']
    ci_durations[category] = {
        'lower': bootstrap_result['ci_lower'], 
        'upper': bootstrap_result['ci_upper']
    }

# Print results
for category in mean_durations:
    print(f"{category}:")
    print(f"  Mean duration: {mean_durations[category]:.4f}")
    print(f"  95% CI: [{ci_durations[category]['lower']:.4f}, {ci_durations[category]['upper']:.4f}]")
    
    
    

    