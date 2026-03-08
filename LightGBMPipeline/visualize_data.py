"""
Data Visualization Script for Driver State Classification

This script creates visualizations to understand the data and model performance.

Usage:
    python visualize_data.py --max-files 10
    python visualize_data.py --max-files 10 --save-dir plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from RandomForest.DataProcessor import find_pairs, sync_all, LINEAR

from FeatureEngineer import FeatureEngineer
from AbsoluteThresholdLabeler import AbsoluteThresholdLabeler
from DriverStateClassifier import DriverStateClassifier

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_process_data(max_files=None, interpolation_mode=LINEAR):
    """Load and process data for visualization"""
    print("Loading data...")
    
    max_pairs = max_files if max_files is not None else 1000
    all_pairs = find_pairs(max_pairs=max_pairs)
    
    combined_data = []
    session_ids = []
    
    for idx, pair_data in enumerate(all_pairs):
        if len(pair_data) == 3:
            df_ego, df_phys, df_surround = pair_data
        elif len(pair_data) == 5:
            df_ego, df_phys, df_surround, participant_id, file_id = pair_data
        else:
            continue
        
        try:
            df_synced = sync_all(df_ego, df_phys, df_surround, method=interpolation_mode)
            df_synced['session_id'] = idx
            combined_data.append(df_synced)
            session_ids.extend([idx] * len(df_synced))
        except Exception as e:
            print(f"Error processing session {idx + 1}: {e}")
            continue
    
    if not combined_data:
        raise ValueError("No data was successfully loaded!")
    
    df = pd.concat(combined_data, ignore_index=True)
    session_ids = np.array(session_ids)
    
    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df, window_size=50)
    features['session_id'] = session_ids
    
    # Generate labels
    print("Generating labels...")
    labeler = AbsoluteThresholdLabeler(use_data_driven_thresholds=True)
    labels = labeler.generate_labels(features)
    features['label'] = labels
    features['label_name'] = pd.Series(labels).map({0: 'Attentive', 1: 'Inattentive', 2: 'Aggressive'})
    
    return df, features, labels, session_ids


def plot_raw_data_distributions(df, save_dir=None):
    """Plot distributions of raw sensor data"""
    print("Creating raw data distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Raw Sensor Data Distributions', fontsize=16, fontweight='bold')
    
    # Acceleration
    if 'oveBodyAccelerationLongitudinalX' in df.columns:
        axes[0, 0].hist(df['oveBodyAccelerationLongitudinalX'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Longitudinal Acceleration')
        axes[0, 0].set_xlabel('Acceleration (m/s²)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
        axes[0, 0].legend()
    
    # Steering
    if 'steeringWheelAngle' in df.columns:
        axes[0, 1].hist(df['steeringWheelAngle'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Steering Wheel Angle')
        axes[0, 1].set_xlabel('Angle (rad)')
        axes[0, 1].set_ylabel('Frequency')
    
    # Speed
    if 'oveBodyVelocityX' in df.columns and 'oveBodyVelocityY' in df.columns:
        speed = np.sqrt(df['oveBodyVelocityX']**2 + df['oveBodyVelocityY']**2)
        axes[0, 2].hist(speed.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Vehicle Speed')
        axes[0, 2].set_xlabel('Speed (m/s)')
        axes[0, 2].set_ylabel('Frequency')
    
    # Heart Rate
    if 'heartRate' in df.columns:
        axes[1, 0].hist(df['heartRate'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title('Heart Rate')
        axes[1, 0].set_xlabel('Heart Rate (BPM)')
        axes[1, 0].set_ylabel('Frequency')
    
    # Gaze Heading
    if 'openxrGazeHeading' in df.columns:
        axes[1, 1].hist(df['openxrGazeHeading'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_title('Gaze Heading')
        axes[1, 1].set_xlabel('Gaze Angle (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(0, color='red', linestyle='--', label='Forward')
        axes[1, 1].legend()
    
    # Eyelid Opening
    if 'varjoEyelidOpening' in df.columns:
        axes[1, 2].hist(df['varjoEyelidOpening'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 2].set_title('Eyelid Opening')
        axes[1, 2].set_xlabel('Opening Ratio')
        axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'raw_data_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'raw_data_distributions.png')}")
    else:
        plt.show()
    plt.close()


def plot_feature_distributions(features, save_dir=None):
    """Plot distributions of engineered features by label"""
    print("Creating feature distribution plots...")
    
    # Select key features
    key_features = [
        'accel_long_std', 'jerk_long_max', 'steering_rate',
        'lateral_pos_std', 'gaze_off_road_ratio', 'hr_mean'
    ]
    
    available_features = [f for f in key_features if f in features.columns]
    
    if not available_features:
        print("No key features available for plotting")
        return
    
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Engineered Features by Driver State', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        
        for label_val, label_name, color in [(0, 'Attentive', 'green'), 
                                             (1, 'Inattentive', 'orange'), 
                                             (2, 'Aggressive', 'red')]:
            data = features[features['label'] == label_val][feature].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.5, label=label_name, color=color, edgecolor='black')
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'feature_distributions.png')}")
    else:
        plt.show()
    plt.close()


def plot_label_distribution(labels, save_dir=None):
    """Plot label distribution"""
    print("Creating label distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    label_counts = pd.Series(labels).value_counts().sort_index()
    label_names = ['Attentive', 'Inattentive', 'Aggressive']
    colors = ['green', 'orange', 'red']
    
    bars = ax1.bar(label_names, [label_counts.get(0, 0), label_counts.get(1, 0), label_counts.get(2, 0)], 
                   color=colors, edgecolor='black', alpha=0.7)
    ax1.set_title('Label Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Driver State')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    counts = [label_counts.get(0, 0), label_counts.get(1, 0), label_counts.get(2, 0)]
    percentages = [c / len(labels) * 100 for c in counts]
    
    ax2.pie(counts, labels=[f'{name}\n({p:.1f}%)' for name, p in zip(label_names, percentages)],
            colors=colors, autopct='', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Label Distribution (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'label_distribution.png')}")
    else:
        plt.show()
    plt.close()


def plot_feature_correlation(features, save_dir=None):
    """Plot correlation matrix of key features"""
    print("Creating feature correlation plot...")
    
    # Select key features
    key_features = [
        'accel_long_std', 'jerk_long_max', 'steering_rate', 'throttle_changes',
        'lateral_pos_std', 'gaze_off_road_ratio', 'ndrt_error_rate',
        'hr_mean', 'hrv_rmssd', 'control_smoothness'
    ]
    
    available_features = [f for f in key_features if f in features.columns]
    
    if len(available_features) < 3:
        print("Not enough features for correlation matrix")
        return
    
    corr_data = features[available_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'feature_correlation.png')}")
    else:
        plt.show()
    plt.close()


def plot_time_series_example(features, session_id=0, save_dir=None):
    """Plot time series example for one session"""
    print(f"Creating time series plot for session {session_id}...")
    
    session_data = features[features['session_id'] == session_id].copy()
    
    if len(session_data) == 0:
        print(f"No data for session {session_id}")
        return
    
    # Sample every 10th point for visualization (too many points otherwise)
    sample_data = session_data.iloc[::10].copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Time Series Example - Session {session_id}', fontsize=16, fontweight='bold')
    
    time = sample_data['time'].values
    
    # Plot 1: Acceleration and Jerk
    if 'accel_long_std' in sample_data.columns:
        axes[0].plot(time, sample_data['accel_long_std'], label='Accel Std', alpha=0.7)
    if 'jerk_long_max' in sample_data.columns:
        axes[0].plot(time, sample_data['jerk_long_max'], label='Jerk Max', alpha=0.7)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Aggressive Indicators')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Gaze and Lane Position
    if 'gaze_off_road_ratio' in sample_data.columns:
        axes[1].plot(time, sample_data['gaze_off_road_ratio'], label='Gaze Off Road', alpha=0.7, color='orange')
    if 'lateral_pos_std' in sample_data.columns:
        axes[1].plot(time, sample_data['lateral_pos_std'], label='Lateral Pos Std', alpha=0.7, color='purple')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Inattentive Indicators')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Heart Rate
    if 'hr_mean' in sample_data.columns:
        axes[2].plot(time, sample_data['hr_mean'], label='Heart Rate', alpha=0.7, color='red')
    axes[2].set_ylabel('Heart Rate (BPM)')
    axes[2].set_title('Physiological Indicator')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Labels
    label_colors = sample_data['label'].map({0: 'green', 1: 'orange', 2: 'red'})
    axes[3].scatter(time, sample_data['label'], c=label_colors, alpha=0.6, s=10)
    axes[3].set_ylabel('Label')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('Driver State Labels')
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(['Attentive', 'Inattentive', 'Aggressive'])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'time_series_session_{session_id}.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, f'time_series_session_{session_id}.png')}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    """Plot confusion matrix"""
    print("Creating confusion matrix plot...")
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Attentive', 'Inattentive', 'Aggressive'],
                yticklabels=['Attentive', 'Inattentive', 'Aggressive'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'confusion_matrix.png')}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(feature_names, importance_values, save_dir=None, top_n=15):
    """Plot feature importance"""
    print("Creating feature importance plot...")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'feature_importance.png')}")
    else:
        plt.show()
    plt.close()


def plot_model_performance_after_training(classifier, features, labels, session_ids, save_dir=None):
    """Plot model performance metrics"""
    print("Training model and creating performance plots...")
    
    # Train model
    cv_results = classifier.train_with_cross_validation(features, labels, session_ids)
    
    # Train final model
    classifier.train_final_model(features, labels)
    
    # Get predictions
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)
    
    # Create combined plots (keeping original layout for backward compatibility)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Attentive', 'Inattentive', 'Aggressive'],
                yticklabels=['Attentive', 'Inattentive', 'Aggressive'])
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Feature Importance
    if hasattr(classifier.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': classifier.feature_cols,
            'importance': classifier.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        axes[0, 1].barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        axes[0, 1].set_yticks(range(len(importance_df)))
        axes[0, 1].set_yticklabels(importance_df['feature'])
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 15 Feature Importance', fontweight='bold')
        axes[0, 1].invert_yaxis()
    
    # 3. Prediction Probabilities Distribution
    axes[1, 0].hist(probabilities[:, 0], bins=50, alpha=0.5, label='Attentive', color='green', edgecolor='black')
    axes[1, 0].hist(probabilities[:, 1], bins=50, alpha=0.5, label='Inattentive', color='orange', edgecolor='black')
    axes[1, 0].hist(probabilities[:, 2], bins=50, alpha=0.5, label='Aggressive', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. CV Scores
    if 'cv_scores' in cv_results:
        axes[1, 1].bar(range(len(cv_results['cv_scores'])), cv_results['cv_scores'], 
                       color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].axhline(np.mean(cv_results['cv_scores']), color='red', 
                          linestyle='--', label=f"Mean: {np.mean(cv_results['cv_scores']):.3f}")
        axes[1, 1].set_xlabel('CV Fold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Cross-Validation Scores', fontweight='bold')
        axes[1, 1].set_xticks(range(len(cv_results['cv_scores'])))
        axes[1, 1].set_xticklabels([f'Fold {i+1}' for i in range(len(cv_results['cv_scores']))])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'model_performance.png')}")
    else:
        plt.show()
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Visualize Driver State Classification Data')
    parser.add_argument('--max-files', type=int, default=5,
                      help='Maximum number of files to load')
    parser.add_argument('--save-dir', type=str, default='plots',
                      help='Directory to save plots (None = display only)')
    parser.add_argument('--interpolation', type=int, default=1,
                      choices=[0, 1, 2],
                      help='Interpolation mode: 0=MERGE, 1=LINEAR, 2=SPLINE')
    parser.add_argument('--skip-model', action='store_true',
                      help='Skip model training plots (faster)')
    
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Plots will be saved to: {args.save_dir}")
    
    # Load and process data
    df, features, labels, session_ids = load_and_process_data(
        max_files=args.max_files,
        interpolation_mode=args.interpolation
    )
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create visualizations
    plot_raw_data_distributions(df, save_dir=args.save_dir)
    plot_feature_distributions(features, save_dir=args.save_dir)
    plot_label_distribution(labels, save_dir=args.save_dir)
    plot_feature_correlation(features, save_dir=args.save_dir)
    plot_time_series_example(features, session_id=0, save_dir=args.save_dir)
    
    if not args.skip_model:
        print("\nTraining model for performance plots (this may take a while)...")
        classifier = DriverStateClassifier()
        classifier.labeler = AbsoluteThresholdLabeler()
        plot_model_performance_after_training(classifier, features, labels, session_ids, save_dir=args.save_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    
    if args.save_dir:
        print(f"\nAll plots saved to: {args.save_dir}/")
    else:
        print("\nAll plots displayed.")


if __name__ == "__main__":
    main()

