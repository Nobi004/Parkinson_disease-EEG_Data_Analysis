# -*- coding: utf-8 -*-
"""
Scalable Deep Learning Framework for Parkinson's Disease Classification
Enhanced with Reinforcement Learning, Transformer models, and advanced techniques
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
import torch.nn.functional as F

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

# Additional imports for RL
import gym
from gym import spaces
from collections import deque
import random

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


# ============= ENHANCED MODELS SECTION =============

class AttentionLayer(layers.Layer):
    """Custom attention layer for feature weighting"""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        # Compute attention scores
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, tf.expand_dims(self.u, -1)), axis=1)
        # Apply attention weights
        weighted_input = inputs * attention_weights
        return weighted_input


# ============= REINFORCEMENT LEARNING COMPONENTS =============

class PDClassificationEnv(gym.Env):
    """Custom Environment for PD Classification using RL"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super(PDClassificationEnv, self).__init__()
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Define action and observation spaces
        # Actions: hyperparameter choices (learning rate, dropout, architecture)
        self.action_space = spaces.Discrete(8)  # 8 different configurations
        
        # Observation: current model performance metrics
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 20
        self.best_accuracy = 0
        
    def reset(self):
        self.current_step = 0
        self.best_accuracy = 0
        # Return initial observation
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        # Map action to hyperparameters
        configs = [
            {'lr': 0.001, 'dropout': 0.2, 'layers': [256, 128, 64]},
            {'lr': 0.001, 'dropout': 0.3, 'layers': [512, 256, 128]},
            {'lr': 0.0001, 'dropout': 0.3, 'layers': [256, 128]},
            {'lr': 0.001, 'dropout': 0.4, 'layers': [512, 256, 128, 64]},
            {'lr': 0.0005, 'dropout': 0.2, 'layers': [384, 192, 96]},
            {'lr': 0.001, 'dropout': 0.5, 'layers': [256, 256, 128]},
            {'lr': 0.0001, 'dropout': 0.3, 'layers': [1024, 512, 256]},
            {'lr': 0.001, 'dropout': 0.25, 'layers': [512, 384, 256, 128]}
        ]
        
        config = configs[action]
        
        # Train small model with selected config
        model = self._create_model(config)
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Get performance metrics
        val_loss, val_acc, val_prec, val_rec = model.evaluate(
            self.X_val, self.y_val, verbose=0
        )
        
        # Calculate reward
        improvement = val_acc - self.best_accuracy
        reward = improvement * 10  # Scale reward
        
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            reward += 1  # Bonus for improvement
        
        # Create observation
        obs = np.array([val_acc, val_prec, val_rec, val_loss, 
                       self.current_step/self.max_steps], dtype=np.float32)
        
        done = self.current_step >= self.max_steps
        
        return obs, reward, done, {'config': config, 'accuracy': val_acc}
    
    def _create_model(self, config):
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Dense(config['layers'][0], activation='relu', 
                              input_shape=(self.X_train.shape[1],)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config['dropout']))
        
        # Hidden layers
        for units in config['layers'][1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(config['dropout']))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['lr']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model


class DQNAgent:
    """Deep Q-Network Agent for hyperparameter optimization"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Neural network for Q-value approximation"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DeepLearningModels:
    """Enhanced collection of deep learning models including advanced architectures"""
    
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
    
    def create_attention_mlp(self) -> keras.Model:
        """MLP with attention mechanism"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Feature extraction
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Apply attention
        attention_output = AttentionLayer(128)(x)
        
        # Continue processing
        x = layers.Dense(128, activation='relu')(attention_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_cnn_1d(self) -> keras.Model:
        """1D CNN for feature extraction"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Reshape for 1D convolution
        x = layers.Reshape((self.input_dim, 1))(inputs)
        
        # Conv blocks
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_lstm_model(self) -> keras.Model:
        """LSTM model for sequential pattern recognition"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Reshape for LSTM
        x = layers.Reshape((self.input_dim, 1))(inputs)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(32)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_transformer_model(self) -> keras.Model:
        """Transformer-based model"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Initial projection
        x = layers.Dense(256)(inputs)
        x = layers.Reshape((16, 16))(x)  # Reshape to sequence format
        
        # Positional encoding
        positions = tf.range(start=0, limit=16, delta=1)
        position_embedding = layers.Embedding(input_dim=16, output_dim=16)(positions)
        x = x + position_embedding
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(64, activation='relu')(x)
        ff_output = layers.Dense(16)(ff_output)
        
        # Add & Norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_ensemble_model(self, base_models: List[keras.Model]) -> keras.Model:
        """Ensemble model combining multiple base models"""
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Get predictions from all base models
        outputs = []
        for i, base_model in enumerate(base_models):
            # Make base model non-trainable
            base_model.trainable = False
            # Get base model output
            base_output = base_model(inputs)
            outputs.append(base_output)
        
        # Combine predictions
        if len(outputs) > 1:
            combined = layers.Concatenate()(outputs)
        else:
            combined = outputs[0]
        
        # Meta-learner
        x = layers.Dense(32, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        final_output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=final_output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_autoencoder_classifier(self, encoding_dim: int = 64) -> Tuple[keras.Model, keras.Model]:
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


class AdvancedModelTrainer:
    """Enhanced model training with RL-based hyperparameter optimization"""
    
    def __init__(self):
        self.results = {}
        self.trained_models = {}
        self.best_configs = {}
        
    def get_callbacks(self, model_name: str) -> List[callbacks.Callback]:
        """Get enhanced training callbacks"""
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
            ),
            callbacks.TensorBoard(
                log_dir=f'logs/{model_name}',
                histogram_freq=1,
                write_graph=True
            )
        ]
    
    def train_with_dqn_optimization(self, model_creator, X_train: np.ndarray, 
                                   y_train: np.ndarray, X_val: np.ndarray, 
                                   y_val: np.ndarray, model_name: str) -> Dict:
        """Train model with DQN-based hyperparameter optimization"""
        
        print(f"\nTraining {model_name} with DQN optimization...")
        
        # Create environment and agent
        env = PDClassificationEnv(X_train, y_train, X_val, y_val)
        agent = DQNAgent(state_size=5, action_size=8)
        
        # Training loop
        best_config = None
        best_accuracy = 0
        episodes = 10
        
        for episode in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, 5])
            
            for step in range(env.max_steps):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, 5])
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                if info['accuracy'] > best_accuracy:
                    best_accuracy = info['accuracy']
                    best_config = info['config']
                
                if done:
                    print(f"Episode {episode+1}/{episodes}, Best Accuracy: {best_accuracy:.4f}")
                    break
                
                if len(agent.memory) > 32:
                    agent.replay(32)
            
            # Update target network
            if episode % 5 == 0:
                agent.update_target_model()
        
        # Train final model with best config
        print(f"Training final model with best configuration...")
        final_model = env._create_model(best_config)
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=self.get_callbacks(model_name),
            verbose=1
        )
        
        self.trained_models[model_name] = final_model
        self.best_configs[model_name] = best_config
        
        return history
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str,
                   epochs: int = 100, batch_size: int = 32) -> Dict:
        """Standard model training"""
        
        print(f"\nTraining {model_name}...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            verbose=1,
            class_weight={0: 1.0, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}  # Handle imbalance
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
        """Perform stratified cross-validation"""
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
                verbose=0,
                class_weight={0: 1.0, 1: len(y_fold_train[y_fold_train==0])/len(y_fold_train[y_fold_train==1])}
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


class EnhancedVisualizer:
    """Enhanced visualization with advanced plots"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        
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
        """Enhanced model comparison visualization"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar plot
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap
        ax2 = axes[0, 1]
        heatmap_data = []
        for model in model_names:
            heatmap_data.append([results[model][metric] for metric in metrics])
        
        sns.heatmap(heatmap_data, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=model_names,
                   annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Performance Heatmap')
        
        # 3. Radar chart
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for model in model_names[:3]:  # Top 3 models
            values = [results[model][metric] for metric in metrics]
            values += values[:1]
            ax3.plot(angles, values, 'o-', linewidth=2, label=model)
            ax3.fill(angles, values, alpha=0.25)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax3.set_ylim(0, 1)
        ax3.set_title('Top 3 Models - Radar Chart')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. Box plot for cross-validation results
        ax4 = axes[1, 1]
        if any('cv_scores' in model for model in results.values() if isinstance(model, dict)):
            cv_data = []
            cv_labels = []
            for model_name, data in results.items():
                if isinstance(data, dict) and 'cv_scores' in data:
                    cv_data.append(data['cv_scores']['accuracy'])
                    cv_labels.append(model_name)
            
            if cv_data:
                ax4.boxplot(cv_data, labels=cv_labels)
                ax4.set_ylabel('Accuracy')
                ax4.set_title('Cross-Validation Results')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No CV data available', ha='center', va='center')
                ax4.set_title('Cross-Validation Results')
        
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
        if n_models == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
            axes[i].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                       save_path: Optional[str] = None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            y_pred_proba = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Main execution class
class EnhancedPDClassificationPipeline:
    """Enhanced scalable pipeline with RL and advanced models"""
    
    def __init__(self, data_path: str = "ds004584-download"):
        self.data_processor = EEGDataProcessor(data_path)
        self.trainer = AdvancedModelTrainer()
        self.visualizer = EnhancedVisualizer()
        self.results = {}
        
    def run_complete_pipeline(self, feature_type: str = 'combined', 
                            test_size: float = 0.2, 
                            perform_cv: bool = True,
                            use_dqn_optimization: bool = True):
        """Run the enhanced classification pipeline"""
        
        print("="*60)
        print("ENHANCED PARKINSON'S DISEASE CLASSIFICATION PIPELINE")
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
        
        # Try multiple scaling methods
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        best_scaler = 'standard'  # Default
        scaler = scalers[best_scaler]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Step 3: Model creation and training
        print("\n3. Creating and training enhanced models...")
        model_factory = DeepLearningModels(X_train_scaled.shape[1])
        
        models_to_train = {
            'Basic_MLP': model_factory.create_basic_mlp,
            'Deep_MLP': model_factory.create_deep_mlp,
            'Attention_MLP': model_factory.create_attention_mlp,
            'CNN_1D': model_factory.create_cnn_1d,
            'LSTM': model_factory.create_lstm_model,
            'Transformer': model_factory.create_transformer_model,
        }
        
        histories = {}
        
        # Train individual models
        for model_name, model_creator in models_to_train.items():
            if use_dqn_optimization and model_name in ['Basic_MLP', 'Deep_MLP']:
                # Use DQN optimization for some models
                history = self.trainer.train_with_dqn_optimization(
                    model_creator, X_train_split, y_train_split,
                    X_val_split, y_val_split, model_name
                )
            else:
                # Standard training
                model = model_creator()
                history = self.trainer.train_model(
                    model, X_train_split, y_train_split, 
                    X_val_split, y_val_split, model_name
                )
            histories[model_name] = history
        
        # Create ensemble model
        print("\n4. Creating ensemble model...")
        base_models = [self.trainer.trained_models[name] for name in ['Basic_MLP', 'Deep_MLP', 'Attention_MLP']]
        ensemble_model = model_factory.create_ensemble_model(base_models)
        ensemble_history = self.trainer.train_model(
            ensemble_model, X_train_split, y_train_split,
            X_val_split, y_val_split, 'Ensemble'
        )
        histories['Ensemble'] = ensemble_history
        
        # Step 5: Evaluation
        print("\n5. Evaluating models...")
        all_models = list(models_to_train.keys()) + ['Ensemble']
        
        for model_name in all_models:
            model = self.trainer.trained_models[model_name]
            metrics = self.trainer.evaluate_model(model, X_test_scaled, y_test, model_name)
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Step 6: Cross-validation (optional)
        if perform_cv:
            print("\n6. Performing cross-validation...")
            cv_results = {}
            for model_name, model_creator in list(models_to_train.items())[:3]:  # CV for top 3 models
                cv_result = self.trainer.cross_validate_model(
                    model_creator, X_train_scaled, y_train
                )
                cv_results[model_name] = cv_result
                print(f"\n{model_name} CV Results:")
                for metric, value in cv_result.items():
                    print(f"{metric}: {value:.4f}")
        
        # Step 7: Visualization
        print("\n7. Generating enhanced visualizations...")
        self.visualizer.plot_training_history(histories, 'training_history_enhanced.png')
        self.visualizer.plot_model_comparison(self.trainer.results, 'model_comparison_enhanced.png')
        self.visualizer.plot_confusion_matrices(self.trainer.results, 'confusion_matrices_enhanced.png')
        self.visualizer.plot_roc_curves(self.trainer.trained_models, X_test_scaled, y_test, 'roc_curves.png')
        
        # Step 8: Save results
        self.results = {
            'test_results': self.trainer.results,
            'cv_results': cv_results if perform_cv else None,
            'best_configs': self.trainer.best_configs,
            'dataset_info': {
                'total_samples': len(X),
                'features': X.shape[1],
                'feature_type': feature_type,
                'pd_samples': np.sum(y),
                'control_samples': len(y) - np.sum(y),
                'scaler_used': best_scaler
            }
        }
        
        print("\n" + "="*60)
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.results


# Usage example
if __name__ == "__main__":
    # Initialize and run enhanced pipeline
    pipeline = EnhancedPDClassificationPipeline()
    results = pipeline.run_complete_pipeline(
        feature_type='combined',  # 'spectral', 'time_domain', or 'combined'
        test_size=0.2,
        perform_cv=True,
        use_dqn_optimization=True  # Enable DQN-based hyperparameter optimization
    )
    
    # Print summary
    print("\nFINAL SUMMARY:")
    print("-" * 40)
    best_model = max(results['test_results'].items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"Best F1-Score: {best_model[1]['f1_score']:.4f}")
    print(f"Best Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"Best AUC-ROC: {best_model[1]['auc_roc']:.4f}")
    
    print("\nAll Models Performance:")
    for model_name, metrics in sorted(results['test_results'].items(), 
                                    key=lambda x: x[1]['f1_score'], reverse=True):
        print(f"{model_name:15s} - Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc_roc']:.4f}")