import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import ttest_1samp, ttest_rel
from mne.filter import filter_data
from mne.stats import spatio_temporal_cluster_test, permutation_cluster_test


def apply_tfce_clustering(scores, baseline_scores=None, adjacency=None, 
                         n_permutations=1000, alpha=0.05, connectivity='auto'):
    """
    Apply TFCE clustering for significance testing on 1D or 2D data.
    
    Parameters:
    -----------
    scores : array, shape (n_subjects, ...) 
        Main scores to test. Can be 1D (n_subjects, n_times) or 2D (n_subjects, n_times, n_times)
    baseline_scores : array, shape (n_subjects, ...) or None
        Baseline scores for comparison. If None, tests against zero.
    adjacency : scipy.sparse matrix or None
        Adjacency matrix for clustering. If None, will be auto-generated.
    n_permutations : int
        Number of permutations for the test
    alpha : float
        Significance threshold
    connectivity : str, 'auto', '1d', '2d_4', '2d_8'
        Type of connectivity to use if adjacency is None
        
    Returns:
    --------
    mask : array, shape (...) 
        Boolean mask where True = non-significant, False = significant
    cluster_info : dict
        Information about detected clusters
    """
    from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
    from scipy.sparse import csr_matrix
    from scipy.stats import ttest_1samp, ttest_rel
    
    # Determine data dimensionality
    if scores.ndim == 2:
        data_type = '1d'
        n_subjects, n_features = scores.shape
        feature_shape = (n_features,)
    elif scores.ndim == 3:
        data_type = '2d'
        n_subjects, n_dim1, n_dim2 = scores.shape
        n_features = n_dim1 * n_dim2
        feature_shape = (n_dim1, n_dim2)
    else:
        raise ValueError(f"Unsupported data shape: {scores.shape}")
    
    print(f"TFCE clustering on {data_type} data: {scores.shape}")
    
    def create_1d_adjacency(n_points):
        """Create adjacency for 1D line (temporal connectivity)."""
        rows, cols, data = [], [], []
        for i in range(n_points):
            for neighbor in [i-1, i+1]:
                if 0 <= neighbor < n_points:
                    rows.extend([i, neighbor])
                    cols.extend([neighbor, i])
                    data.extend([1, 1])
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def create_2d_adjacency(n_rows, n_cols, connectivity_type='8'):
        """Create adjacency for 2D grid."""
        n_vertices = n_rows * n_cols
        rows, cols, data = [], [], []
        
        if connectivity_type == '4':
            neighbors_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity_type == '8':
            neighbors_delta = [(-1, -1), (-1, 0), (-1, 1), 
                             (0, -1), (0, 1),
                             (1, -1), (1, 0), (1, 1)]
        
        for i in range(n_rows):
            for j in range(n_cols):
                current = i * n_cols + j
                for di, dj in neighbors_delta:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_rows and 0 <= nj < n_cols:
                        neighbor = ni * n_cols + nj
                        rows.extend([current, neighbor])
                        cols.extend([neighbor, current])
                        data.extend([1, 1])
        
        return csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    # Create adjacency matrix if not provided
    if adjacency is None:
        if connectivity == 'auto':
            if data_type == '1d':
                adjacency = create_1d_adjacency(n_features)
            else:  # 2d
                adjacency = create_2d_adjacency(feature_shape[0], feature_shape[1], '8')
        elif connectivity == '1d':
            adjacency = create_1d_adjacency(n_features)
        elif connectivity == '2d_4':
            adjacency = create_2d_adjacency(feature_shape[0], feature_shape[1], '4')
        elif connectivity == '2d_8':
            adjacency = create_2d_adjacency(feature_shape[0], feature_shape[1], '8')
    
    # Reshape data for clustering
    X = scores.reshape(n_subjects, n_features)
    
    # Prepare data for testing
    if baseline_scores is not None:
        # Test difference against baseline
        X_baseline = baseline_scores.reshape(n_subjects, n_features)
        X_diff = X - X_baseline
        test_data = [X_diff]
        test_type = "paired"
    else:
        # Test against zero
        test_data = [X]
        test_type = "one-sample"
    
    try:
        # Run cluster permutation test
        if data_type == '1d':
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                test_data,
                adjacency=adjacency,
                n_permutations=n_permutations,
                threshold=None,  # TFCE approach
                n_jobs=-2,  # Use all but one CPU
                seed=42,
                out_type='mask'
            )
        else:  # 2d
            T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
                test_data,
                adjacency=adjacency,
                n_permutations=n_permutations,
                threshold=None,  # TFCE approach
                n_jobs=-2,  # Use all but one CPU
                seed=42
            )
        
        # Create significance mask
        mask = np.ones(feature_shape, dtype=bool)
        significant_clusters = []
        
        if len(clusters) > 0:
            for cluster_idx, cluster_p in enumerate(cluster_p_values):
                if cluster_p < alpha:
                    if data_type == '1d':
                        cluster_mask = clusters[cluster_idx]
                        mask[cluster_mask] = False
                    else:  # 2d
                        if isinstance(clusters[cluster_idx], tuple):
                            # spatio_temporal_cluster_test returns (space, time) tuple
                            cluster_mask = clusters[cluster_idx][0]
                        else:
                            cluster_mask = clusters[cluster_idx]
                        cluster_2d = cluster_mask.reshape(feature_shape)
                        mask[cluster_2d] = False
                    
                    significant_clusters.append({
                        'cluster_id': cluster_idx,
                        'p_value': cluster_p,
                        'size': np.sum(~cluster_mask if data_type == '1d' else ~cluster_2d)
                    })
        
        cluster_info = {
            'test_type': test_type,
            'n_clusters_total': len(clusters),
            'n_clusters_significant': len(significant_clusters),
            'significant_clusters': significant_clusters,
            'alpha': alpha
        }
        
        print(f"TFCE ({test_type}): {len(clusters)} clusters found, "
              f"{len(significant_clusters)} significant (p < {alpha})")
        
        return mask, cluster_info
        
    except Exception as e:
        print(f"TFCE clustering failed ({e}), falling back to uncorrected testing")
        
        # Fallback to uncorrected pixel/timepoint-wise testing
        mask = np.ones(feature_shape, dtype=bool)
        p_values = np.zeros(feature_shape)
        
        if baseline_scores is not None:
            X_baseline = baseline_scores.reshape(n_subjects, n_features)
            for idx in range(n_features):
                _, p_val = ttest_rel(X[:, idx], X_baseline[:, idx])
                flat_coord = np.unravel_index(idx, feature_shape)
                p_values[flat_coord] = p_val
                if p_val < alpha:
                    mask[flat_coord] = False
        else:
            for idx in range(n_features):
                _, p_val = ttest_1samp(X[:, idx], 0)
                flat_coord = np.unravel_index(idx, feature_shape)
                p_values[flat_coord] = p_val
                if p_val < alpha:
                    mask[flat_coord] = False
        
        cluster_info = {
            'test_type': 'uncorrected_fallback',
            'n_significant': np.sum(~mask),
            'alpha': alpha
        }
        
        return mask, cluster_info
