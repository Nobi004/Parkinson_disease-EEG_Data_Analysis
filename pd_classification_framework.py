# -*- coding: utf-8 -*-
"""
Scalable Deep Learning Framework for Parkinson's Disease Classification
This framework provides multiple deep learning approaches for EEG-based PD classification
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Deep Learning and ML imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Scientific computing and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import mne

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

class EEGDataProcessor:
    """Scalable EEG data processing class"""
    
    def __init__(self, data_path: str = "ds004584-download"):
        self.data_path = Path(data_path)
        self.scaler = None
        self.common_channels = None
        self.brain_wave_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'low_gamma': (30, 50),
            'high_gamma': (50, 100)
        }
    
    def gather_file_paths(self) -> List[Path]:
        """Gather all .set file paths"""
        return list(self.data_path.glob("**/*.set"))
    
    def find_common_channels(self, file_paths: List[Path]) -> List[str]:
        """Find channels common to all recordings"""
        if self.common_channels is not None:
            return self.common_channels
            
        for idx, filepath in enumerate(file_paths):
            raw = mne.io.read_raw_eeglab(filepath, verbose=False)
            ch_names = raw.ch_names
            if idx == 0:
                self.common_channels = set(ch_names)
            else:
                self.common_channels = self.common_channels.intersection(set(ch_names))
        
        self.common_channels = list(self.common_channels)
        return self.common_channels
    
    def extract_spectral_features(self, filepath: Path, channels: List[str]) -> np.ndarray:
        """Extract comprehensive spectral features from EEG data"""
        raw = mne.io.read_raw_eeglab(filepath, verbose=False)
        spectrum = raw.compute_psd(picks=channels, n_jobs=-1, verbose=False)
        data, freqs = spectrum.get_data(return_freqs=True)
        
        features = []
        
        # Band power features
        for band, (low, high) in self.brain_wave_bands.items():
            indices = np.where((freqs >= low) & (freqs <= high))[0]
            if len(indices) > 0:
                band_power = np.sum(data[:, indices], axis=1)
                features.extend([
                    np.mean(band_power),  # Mean band power
                    np.std(band_power),   # Std band power
                    np.max(band_power),   # Max band power
                    np.median(band_power) # Median band power
                ])
        
        # Statistical features across all frequencies
        features.extend([
            np.mean(data),  # Overall mean power
            np.std(data),   # Overall std power
            np.var(data),   # Overall variance
            np.max(data),   # Peak power
            np.min(data),   # Min power
        ])
        
        # Spectral edge frequency (95% of power)
        cumulative_power = np.cumsum(np.mean(data, axis=0))
        total_power = cumulative_power[-1]
        edge_freq_idx = np.where(cumulative_power >= 0.95 * total_power)[0]
        if len(edge_freq_idx) > 0:
            features.append(freqs[edge_freq_idx[0]])
        else:
            features.append(freqs[-1])
        
        # Dominant frequency
        mean_power = np.mean(data, axis=0)
        dominant_freq_idx = np.argmax(mean_power)
        features.append(freqs[dominant_freq_idx])
        
        return np.array(features)
    
    def extract_time_domain_features(self, filepath: Path, channels: List[str]) -> np.ndarray:
        """Extract time domain features"""
        raw = mne.io.read_raw_eeglab(filepath, verbose=False)
        data = raw.get_data(picks=channels)
        
        features = []
        
        for ch_data in data:
            features.extend([
                np.mean(ch_data),           # Mean
                np.std(ch_data),            # Standard deviation
                np.var(ch_data),            # Variance
                np.max(ch_data),            # Maximum
                np.min(ch_data),            # Minimum
                np.median(ch_data),         # Median
                np.percentile(ch_data, 25), # Q1
                np.percentile(ch_data, 75), # Q3
                np.ptp(ch_data),           # Peak-to-peak
                np.mean(np.abs(ch_data)),  # Mean absolute value
            ])
        
        return np.array(features)
    
    def prepare_dataset(self, feature_type: str = 'spectral') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare complete dataset with features and labels"""
        file_paths = self.gather_file_paths()
        common_channels = self.find_common_channels(file_paths)
        
        # Load subject info
        subject_info = pd.read_csv(self.data_path / "participants.tsv", sep='\t')
        
        X = []
        y = []
        participant_ids = []
        
        print(f"Processing {len(file_paths)} files...")
        
        for filepath in file_paths:
            try:
                participant_id = filepath.stem[:7]
                
                # Extract features based on type
                if feature_type == 'spectral':
                    features = self.extract_spectral_features(filepath, common_channels)
                elif feature_type == 'time_domain':
                    features = self.extract_time_domain_features(filepath, common_channels)
                elif feature_type == 'combined':
                    spectral_features = self.extract_spectral_features(filepath, common_channels)
                    time_features = self.extract_time_domain_features(filepath, common_channels)
                    features = np.concatenate([spectral_features, time_features])
                
                # Get label
                mask = subject_info.participant_id == participant_id
                if mask.sum() > 0:
                    label = subject_info.loc[mask, 'GROUP'].values[0]
                    X.append(features)
                    y.append(1 if label == 'PD' else 0)
                    participant_ids.append(participant_id)
                    
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        return np.array(X), np.array(y), participant_ids


class DeepLearningModels:
    """Collection of scalable deep learning models for PD classification"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.models = {}
        
    def create_basic_mlp(self, dropout_rate: float = 0.3) -> keras.Model:
        """Basic Multi-Layer Perceptron"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate/2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout_rate/2),
            
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_deep_mlp(self, dropout_rate: float = 0.4) -> keras.Model:
        """Deeper MLP with residual connections"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # First block
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Second block with residual
        residual = layers.Dense(256)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Third block
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Fourth block
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate/2)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_autoencoder_classifier(self, encoding_dim: int = 64) -> keras.Model:
        """Autoencoder-based classifier"""
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(256, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.3)(encoded)
        encoded = layers.Dense(128, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(128, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(256, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Classifier from encoded representation
        classifier = layers.Dense(32, activation='relu')(encoded)
        classifier = layers.Dropout(0.2)(classifier)
        classifier = layers.Dense(1, activation='sigmoid')(classifier)
        
        # Create models
        autoencoder = models.Model(input_layer, decoded)
        classifier_model = models.Model(input_layer, classifier)
        
        # Compile
        autoencoder.compile(optimizer='adam', loss='mse')
        classifier_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return autoencoder, classifier_model


class ModelTrainer:
    """Scalable model training and evaluation class"""
    
    def __init__(self):
        self.results = {}
        self.trained_models = {}
        
    def get_callbacks(self, model_name: str) -> List[callbacks.Callback]:
        """Get training callbacks"""
        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_{model_name}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str,
                   epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train a single model"""
        
        print(f"\nTraining {model_name}...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            verbose=1
        )
        
        self.trained_models[model_name] = model
        return history
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str) -> Dict:
        """Comprehensive model evaluation"""
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def cross_validate_model(self, model_creator, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5) -> Dict:
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_creator()
            model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=50,
                batch_size=32,
                callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_fold_val, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            cv_scores['precision'].append(precision_score(y_fold_val, y_pred))
            cv_scores['recall'].append(recall_score(y_fold_val, y_pred))
            cv_scores['f1'].append(f1_score(y_fold_val, y_pred))
            cv_scores['auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        return cv_results


class Visualizer:
    """Comprehensive visualization class"""
    
    def __init__(self):
        plt.style.use('default')
        
    def plot_training_history(self, histories: Dict, save_path: Optional[str] = None):
        """Plot training histories for multiple models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison', fontsize=16)
        
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            for model_name, history in histories.items():
                if metric in history.history:
                    ax.plot(history.history[metric], label=f'{model_name} - Train')
                    ax.plot(history.history[f'val_{metric}'], label=f'{model_name} - Val', linestyle='--')
            
            ax.set_title(f'{metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Compare model performance"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(model_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Heatmap
        heatmap_data = []
        for model in model_names:
            heatmap_data.append([results[model][metric] for metric in metrics])
        
        sns.heatmap(heatmap_data, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=model_names,
                   annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Performance Heatmap')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results: Dict, save_path: Optional[str] = None):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        # Always flatten axes for consistent indexing
        if n_models == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, (model_name, metrics) in enumerate(results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main execution class
class PDClassificationPipeline:
    """Complete scalable pipeline for PD classification"""
    
    def __init__(self, data_path: str = "ds004584-download"):
        self.data_processor = EEGDataProcessor(data_path)
        self.trainer = ModelTrainer()
        self.visualizer = Visualizer()
        self.results = {}
        
    def run_complete_pipeline(self, feature_type: str = 'combined', 
                            test_size: float = 0.2, 
                            perform_cv: bool = True):
        """Run the complete classification pipeline"""
        
        print("="*60)
        print("PARKINSON'S DISEASE CLASSIFICATION PIPELINE")
        print("="*60)
        
        # Step 1: Data preparation
        print("\n1. Preparing dataset...")
        X, y, participant_ids = self.data_processor.prepare_dataset(feature_type)
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: PD={np.sum(y)}, Control={len(y)-np.sum(y)}")
        
        # Step 2: Data splitting and scaling
        print("\n2. Splitting and scaling data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Step 3: Model creation and training
        print("\n3. Creating and training models...")
        model_factory = DeepLearningModels(X_train_scaled.shape[1])
        
        models_to_train = {
            'Basic_MLP': model_factory.create_basic_mlp,
            'Deep_MLP': model_factory.create_deep_mlp,
        }
        
        histories = {}
        
        for model_name, model_creator in models_to_train.items():
            model = model_creator()
            history = self.trainer.train_model(
                model, X_train_split, y_train_split, 
                X_val_split, y_val_split, model_name
            )
            histories[model_name] = history
        
        # Step 4: Evaluation
        print("\n4. Evaluating models...")
        for model_name in models_to_train.keys():
            model = self.trainer.trained_models[model_name]
            metrics = self.trainer.evaluate_model(model, X_test_scaled, y_test, model_name)
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Step 5: Cross-validation (optional)
        if perform_cv:
            print("\n5. Performing cross-validation...")
            cv_results = {}
            for model_name, model_creator in models_to_train.items():
                cv_result = self.trainer.cross_validate_model(
                    model_creator, X_train_scaled, y_train
                )
                cv_results[model_name] = cv_result
                print(f"\n{model_name} CV Results:")
                for metric, value in cv_result.items():
                    print(f"{metric}: {value:.4f}")
        
        # Step 6: Visualization
        print("\n6. Generating visualizations...")
        self.visualizer.plot_training_history(histories, 'training_history.png')
        self.visualizer.plot_model_comparison(self.trainer.results, 'model_comparison.png')
        self.visualizer.plot_confusion_matrices(self.trainer.results, 'confusion_matrices.png')
        
        # Step 7: Save results
        self.results = {
            'test_results': self.trainer.results,
            'cv_results': cv_results if perform_cv else None,
            'dataset_info': {
                'total_samples': len(X),
                'features': X.shape[1],
                'feature_type': feature_type,
                'pd_samples': np.sum(y),
                'control_samples': len(y) - np.sum(y)
            }
        }
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.results


# Usage example
if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = PDClassificationPipeline()
    results = pipeline.run_complete_pipeline(
        feature_type='combined',  # 'spectral', 'time_domain', or 'combined'
        test_size=0.2,
        perform_cv=True
    )
    
    # Print summary
    print("\nFINAL SUMMARY:")
    print("-" * 40)
    for model_name, metrics in results['test_results'].items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
