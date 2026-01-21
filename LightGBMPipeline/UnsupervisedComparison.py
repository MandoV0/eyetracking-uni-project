"""
Unsupervised Learning: GMM Clustering for Driver State Classification

This module implements Gaussian Mixture Model (GMM) clustering to discover 
driver states without using labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment
import os


def _map_clusters_to_labels(cluster_labels, true_labels, n_clusters=3):
    """Map cluster assignments to true labels using Hungarian algorithm."""
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) == 0:
            continue
        
        for label_id in range(n_clusters):
            cost_matrix[cluster_id, label_id] = np.sum(
                (cluster_labels == cluster_id) & (true_labels == label_id)
            )
    
    row_indices, col_indices = linear_sum_assignment(-cost_matrix)
    
    mapping = {}
    for cluster_id, label_id in zip(row_indices, col_indices):
        mapping[cluster_id] = label_id
    
    mapped_predictions = np.array([mapping.get(c, c) for c in cluster_labels])
    return mapping, mapped_predictions


def run_gmm(features, labels, session_ids, classifier, n_clusters=3, random_state=42,
    test_data=None, test_size=0.2):
    """
    Train and evaluate GMM clustering for driver state classification.
    
    Args:
        features: DataFrame with features
        labels: Array with true labels (for evaluation only)
        session_ids: Array with session IDs (for splitting)
        classifier: DriverStateClassifier instance (for feature prep/scaling)
        n_clusters: Number of clusters (should be 3 for 3 driver states)
        random_state: Random seed
        test_data: Optional tuple (X_test_df, y_test) - if provided, uses this test set
        test_size: Test set size ratio (used if test_data is None)
    """
    print("\n" + "="*70)
    print(" === GMM CLUSTERING ===")
    print("="*70)
    
    # Prepare features
    X = classifier.prepare_features_for_training(features)
    
    # Split data
    if test_data is not None:
        X_test_df, y_test = test_data
        X_test = classifier.prepare_features_for_training(X_test_df)
        
        X_train_scaled = classifier.scaler.fit_transform(X)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        X_train_final = X_train_scaled
        y_train_final = labels
        X_test_final = X_test_scaled
        y_test_final = y_test
    else:
        from sklearn.model_selection import train_test_split
        
        if session_ids is None:
            raise ValueError("Either provide test_data or session_ids for splitting.")
        
        unique_sessions = np.unique(session_ids)
        train_sessions, test_sessions = train_test_split(
            unique_sessions, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        train_mask = np.isin(session_ids, train_sessions)
        test_mask = np.isin(session_ids, test_sessions)
        
        X_train_df = features.loc[train_mask].reset_index(drop=True)
        X_test_df = features.loc[test_mask].reset_index(drop=True)
        y_train_final = labels[train_mask]
        y_test_final = labels[test_mask]
        
        X_train = classifier.prepare_features_for_training(X_train_df)
        X_test = classifier.prepare_features_for_training(X_test_df)
        
        X_train_scaled = classifier.scaler.fit_transform(X_train)
        X_test_scaled = classifier.scaler.transform(X_test)
        
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    
    # Train GMM
    print("\nFitting GMM model on training data...")
    gmm_model = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        max_iter=300,
        covariance_type='full',
    )
    gmm_model.fit(X_train_final)
    
    # Predict clusters
    train_clusters = gmm_model.predict(X_train_final)
    test_clusters = gmm_model.predict(X_test_final)
    
    # Map clusters to labels
    print("Mapping clusters to driver states...")
    cluster_to_label_mapping, train_mapped = _map_clusters_to_labels(
        train_clusters, y_train_final, n_clusters=n_clusters
    )
    test_mapped = np.array([cluster_to_label_mapping.get(c, c) for c in test_clusters])
    
    # Evaluate
    train_acc = accuracy_score(y_train_final, train_mapped)
    test_acc = accuracy_score(y_test_final, test_mapped)
    kappa = cohen_kappa_score(y_test_final, test_mapped)
    ari = adjusted_rand_score(y_test_final, test_clusters)
    nmi = normalized_mutual_info_score(y_test_final, test_clusters)
    
    # Print results
    print(f"\nGMM Train Accuracy: {train_acc:.4f}")
    print(f"GMM Test Accuracy:  {test_acc:.4f}")
    
    print("\nGMM CLASSIFICATION REPORT (test set)")
    print("=" * 60)
    print(classification_report(
        y_test_final, test_mapped,
        target_names=['Attentive', 'Inattentive', 'Aggressive'],
        digits=4
    ))
    
    print("\nGMM CONFUSION MATRIX (test set)")
    print("=" * 60)
    cm = confusion_matrix(y_test_final, test_mapped)
    print(f"{'':>15} {'Attentive':>12} {'Inattentive':>12} {'Aggressive':>12}")
    for i, name in enumerate(['Attentive', 'Inattentive', 'Aggressive']):
        print(f"{name:>15}", end="")
        for val in cm[i]:
            print(f"{val:>12}", end="")
        print()
    
    print(f"\nGMM Cohen's Kappa (test): {kappa:.4f}")
    print(f"GMM Adjusted Rand Index (test): {ari:.4f}")
    print(f"GMM Normalized Mutual Info (test): {nmi:.4f}")
    
    print(f"\nCluster-to-Label Mapping:")
    for cluster_id, label_id in sorted(cluster_to_label_mapping.items()):
        label_name = ['Attentive', 'Inattentive', 'Aggressive'][label_id]
        print(f"  Cluster {cluster_id} -> {label_name} (label {label_id})")
    
    # Return only what's needed for visualization
    return {
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train_final,
        'y_test': y_test_final,
        'train_clusters': train_clusters,
        'test_clusters': test_clusters,
    }


def visualize_clusters(results, save_dir='plots', max_samples=5000, use_tsne=False):
    """Visualize clusters using PCA or t-SNE dimensionality reduction."""
    os.makedirs(save_dir, exist_ok=True)
    
    X_train = results['X_train']
    X_test = results['X_test']
    y_train = results['y_train']
    y_test = results['y_test']
    train_clusters = results['train_clusters']
    test_clusters = results['test_clusters']
    
    # Subsample for visualization
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_viz = X_train[indices]
        y_train_viz = y_train[indices]
        train_clusters_viz = train_clusters[indices]
    else:
        X_train_viz = X_train
        y_train_viz = y_train
        train_clusters_viz = train_clusters
    
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_test_viz = X_test[indices]
        y_test_viz = y_test[indices]
        test_clusters_viz = test_clusters[indices]
    else:
        X_test_viz = X_test
        y_test_viz = y_test
        test_clusters_viz = test_clusters
    
    # Dimensionality reduction
    print(f"\nComputing {'t-SNE' if use_tsne else 'PCA'} for visualization...")
    if use_tsne:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_train_2d = reducer.fit_transform(X_train_viz)
        X_test_2d = reducer.transform(X_test_viz)
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_train_2d = reducer.fit_transform(X_train_viz)
        X_test_2d = reducer.transform(X_test_viz)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_.sum():.2%}")
    
    # Color maps
    label_colors = {0: 'green', 1: 'orange', 2: 'red'}
    label_names = ['Attentive', 'Inattentive', 'Aggressive']
    cluster_colors = {0: 'blue', 1: 'purple', 2: 'cyan'}
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GMM Clustering Visualization', fontsize=16, fontweight='bold')
    
    # Plot 1: Train - True Labels
    ax = axes[0, 0]
    for label_id in [0, 1, 2]:
        mask = y_train_viz == label_id
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
                  c=label_colors[label_id], label=label_names[label_id], 
                  alpha=0.6, s=20)
    ax.set_title('Train Set: True Labels')
    ax.set_xlabel(f'{reducer.__class__.__name__} Component 1')
    ax.set_ylabel(f'{reducer.__class__.__name__} Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Train - Predicted Clusters
    ax = axes[0, 1]
    for cluster_id in [0, 1, 2]:
        mask = train_clusters_viz == cluster_id
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
                  c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}', 
                  alpha=0.6, s=20)
    ax.set_title('Train Set: Predicted Clusters')
    ax.set_xlabel(f'{reducer.__class__.__name__} Component 1')
    ax.set_ylabel(f'{reducer.__class__.__name__} Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Test - True Labels
    ax = axes[1, 0]
    for label_id in [0, 1, 2]:
        mask = y_test_viz == label_id
        ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                  c=label_colors[label_id], label=label_names[label_id], 
                  alpha=0.6, s=20)
    ax.set_title('Test Set: True Labels')
    ax.set_xlabel(f'{reducer.__class__.__name__} Component 1')
    ax.set_ylabel(f'{reducer.__class__.__name__} Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Test - Predicted Clusters
    ax = axes[1, 1]
    for cluster_id in [0, 1, 2]:
        mask = test_clusters_viz == cluster_id
        ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                  c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}', 
                  alpha=0.6, s=20)
    ax.set_title('Test Set: Predicted Clusters')
    ax.set_xlabel(f'{reducer.__class__.__name__} Component 1')
    ax.set_ylabel(f'{reducer.__class__.__name__} Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, 'gmm_clusters_visualization.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")
    plt.close()
