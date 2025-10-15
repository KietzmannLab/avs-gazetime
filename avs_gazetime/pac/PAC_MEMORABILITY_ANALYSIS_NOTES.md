# PAC High vs Low Memorability Analysis - Step-by-Step Guide

## Overview
This analysis compares phase-amplitude coupling (PAC) between fixations on high vs low memorability content, controlling for potential confounds.

---

## Step 1: Data Preparation

### Input Data
- **MEG source-level PAC values** (theta-gamma coupling per source/vertex)
- **Fixation metadata** with memorability scores per fixation
- **ROI labels** mapping sources to brain regions

### Memorability Scores
- Computed using **ResMem model** on fixation crop images
- Each fixation has a memorability score from the pre-trained model
- Scores reflect predicted memorability of visual content at fixation location

---

## Step 2: Stratified Sampling (Controlling for Fixation Duration)

### Why Control for Duration?
- Longer fixations may naturally have different PAC values
- Longer fixations might also have different memorability scores
- This creates a confound: any PAC difference could be due to duration, not memorability

### Stratification Process (`pac_dataloader.py:160-348`)

1. **Define Memorability Groups**
   - Split specification: e.g., `"50/50"` or `"40/40"`
   - Bottom 50%/40% = **Low Memorability** group
   - Top 50%/40% = **High Memorability** group
   - Middle range excluded

2. **Duration Binning**
   - Create 5 duration bins (quintiles) across all fixations
   - Each bin represents 20% of the duration distribution

3. **Stratified Sampling within Each Group**
   - For each memorability group:
     - Sample proportionally from each duration bin
     - Ensures duration distribution is matched between groups
   - Target: Equal number of epochs in high vs low groups

4. **Result**
   - Two groups with:
     - Matched sample sizes
     - Matched duration distributions
     - Different memorability levels

---

## Step 3: PAC Computation per Memorability Group

### For Each Group Separately (`pac_analysis.py`)

1. **Load MEG data** for selected epochs
2. **Band-pass filter**:
   - Theta band (3-8 Hz) for phase
   - Gamma band (40-140 Hz) for amplitude
3. **Compute PAC** using modulation index method
4. **Surrogate Testing**:
   - Generate 200 surrogate datasets (phase shuffle/session-aware/single-cut)
   - Compare true PAC to surrogate distribution
   - Output: **z-scored PAC value** per source

### Output per Subject
- CSV with columns:
  - `channel`: Source/vertex index
  - `pac`: z-scored PAC value
  - `split_group`: "low_mem" or "high_mem"
  - `n_epochs`: Number of epochs in this group
  - `surrogate_style`: Method used

---

## Step 4: Channel-to-ROI Mapping

### Purpose
Map each source (channel/vertex) to anatomical brain regions

### Process (`pac_plotting_and_stats.ipynb:cell-28`)

1. **Load ROI labels** from FreeSurfer for each subject
2. **For each ROI**:
   - Read label file (`.label` format)
   - Extract vertex indices in that label
   - Account for hemisphere offset (right hemisphere indices shifted)
3. **Create mapping**: `channel → (area, hemisphere)`

---

## Step 5: Statistical Analysis

### A. Paired Comparison Across All Sources

**Test**: Paired t-test (high vs low memorability)
- Each source has two PAC values (one per memorability condition)
- Tests whether PAC differs systematically across memorability levels
- Accounts for within-source pairing

### B. Per-ROI Analysis

**For each brain region**:
1. Extract all sources in that ROI
2. Perform paired t-test: `PAC_high vs PAC_low`
3. Apply **FDR correction** across all ROIs
4. Report: effect size (mean difference), t-statistic, corrected p-value

### C. Mixed-Effects Model (Full Model)

**Model**: `PAC ~ mem_high * area * hemi + (1|subject)`

**Components**:
- `mem_high`: Binary predictor (0=low, 1=high memorability)
- `area`: Brain region (categorical)
- `hemi`: Hemisphere (left/right)
- `*`: Full factorial interactions
- `(1|subject)`: Random intercept per subject (accounts for between-subject variability)

**What it tests**:
- Does memorability affect PAC? (main effect)
- Does this effect vary by brain region? (interaction)
- Does it differ between hemispheres? (interaction)

**Advantages**:
- Pools error across all ROIs (more power)
- Tests region-specific effects within single framework
- Proper control for multiple comparisons
- Accounts for hierarchical data structure (sources nested in subjects)

---

## Step 6: Visualization

### Figure 1: Overall Effect
- Point plot: PAC in high vs low memorability groups
- Shows main effect with subject-level variability

### Figure 2: Per-ROI Effects
- Horizontal bar plot: Mean PAC difference per brain region
- Color-coded by FDR significance
- Error bars show standard errors

### Figure 3: Mixed Model Results
- Horizontal bar plot: Memorability effect (β coefficient) per ROI×hemisphere
- Extracted from full model interactions
- Shows region- and hemisphere-specific effects

---

## Key Design Decisions

### Why Stratified Sampling?
Without it, any observed difference could be explained by:
- High mem fixations being longer → more PAC
- Low mem fixations being shorter → less PAC

Stratification ensures both groups have **identical duration distributions**, so effects are due to memorability content, not fixation length.

### Why Mixed Models?
- **Hierarchical structure**: Sources nested within subjects
- **Repeated measures**: Same sources measured in two conditions
- **Between-subject variability**: Random effects capture individual differences
- **Multiple regions**: Fixed effects model region-specific effects

### Why FDR Correction?
- Testing multiple ROIs (or ROI×hemisphere combinations)
- Without correction: inflated false positive rate
- FDR controls expected proportion of false discoveries
- More powerful than Bonferroni (family-wise error rate)

---

## Summary of Results

The analysis provides:
1. **Overall effect**: Is PAC higher for high vs low memorability content?
2. **Regional specificity**: Which brain regions show this effect?
3. **Hemisphere differences**: Does the effect differ between left and right?
4. **Statistical rigor**: Controlled for confounds, corrected for multiple comparisons

All while accounting for:
- Fixation duration confound (stratified sampling)
- Between-subject variability (random effects)
- Within-source pairing (repeated measures)
- Multiple testing (FDR correction)
