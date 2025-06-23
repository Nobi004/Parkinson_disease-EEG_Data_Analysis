import mne
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Attention
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pywt
from scipy.stats import entropy, skew, kurtosis

import pandas as pd
df = pd.read_csv('ds004584-download/participants.tsv', sep='\t')
print(df.columns)
df['GROUP'].map({'PD': 1, 'control': 0})
print(df.head())
# EEG Data Processor with Window Slicing and Advanced Features
class EEGDataProcessor:
    def __init__(self, data_dir, window_size=10, sfreq=256):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.sfreq = sfreq
        self.freq_bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50), (50, 100)]

    def gather_file_paths(self):
        return list(self.data_dir.glob('**/*.set'))

    def load_labels(self, participants_file):
        df = pd.read_csv(participants_file, sep='\t')
        return df['GROUP'].map({'PD': 1, 'control': 0}).values, df['participant_id'].values

    def extract_windowed_data(self, file_paths, window_size_sec=2, step_size_sec=1):
        """
        Extract windowed features from EEG files.
        Returns:
            X: np.ndarray of features
            y: np.ndarray of labels
            groups: list of participant IDs
        """
        import numpy as np
        import mne
        import pandas as pd

        # Load participant info
        participants_path = self.data_dir / "participants.tsv"
        df = pd.read_csv(participants_path, sep='\t')
        # Robust label column detection
        label_col = 'GROUP' if 'GROUP' in df.columns else 'group'
        label_map = {'PD': 1, 'Control': 0, 'control': 0}
        labels = dict(zip(df['participant_id'], df[label_col].map(label_map)))

        X = []
        y = []
        groups = []

        for path in file_paths:
            try:
                raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
                participant_id = path.stem[:7]
                label = labels.get(participant_id)
                if label is None:
                    print(f"Label not found for {participant_id}, skipping.")
                    continue

                sfreq = raw.info['sfreq']
                n_samples = raw.n_times
                max_time = n_samples / sfreq

                start = 0
                while start < max_time:
                    end = start + window_size_sec
                    # Prevent cropping beyond data
                    if end > max_time:
                        end = max_time
                    if start >= end:
                        break

                    try:
                        window_raw = raw.copy().crop(tmin=start, tmax=end)
                        data = window_raw.get_data()
                        # Example: flatten window data as features
                        features = data.flatten()
                        # Handle NaN/Inf
                        if not np.all(np.isfinite(features)):
                            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        X.append(features)
                        y.append(label)
                        groups.append(participant_id)
                    except Exception as e:
                        print(f"Window extraction failed for {participant_id} [{start}-{end}]: {e}")
                    start += step_size_sec

            except Exception as e:
                print(f"Failed to process {path}: {e}")

        return np.array(X), np.array(y), groups

    def extract_advanced_features(self, data):
        # Wavelet Packet Decomposition
        wp = pywt.WaveletPacket(data=data, wavelet='db4', mode='symmetric', maxlevel=4)
        nodes = wp.get_level(4, 'freq')
        wp_coeffs = [node.data for node in nodes]
        wp_features = [np.mean(coeffs) for coeffs in wp_coeffs] + [np.std(coeffs) for coeffs in wp_coeffs]

        # Spectral Entropy
        psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=self.sfreq, fmin=0.5, fmax=40)
        spec_entropy = entropy(psd, axis=1)

        # Hjorth Parameters
        activity = np.var(data, axis=0)
        diff = np.diff(data, axis=0)
        mobility = np.sqrt(np.var(diff, axis=0) / activity)
        complexity = np.sqrt(np.var(np.diff(diff, axis=0), axis=0) / np.var(diff, axis=0)) / mobility

        # Statistical Moments
        skewness = skew(data, axis=0)
        kurt = kurtosis(data, axis=0)

        return np.concatenate([wp_features, spec_entropy, activity, mobility, complexity, skewness, kurt])

# Deep Learning Models
class DeepLearningModels:
    @staticmethod
    def build_1d_cnn(input_shape):
        model = Sequential([
            Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    @staticmethod
    def build_crnn(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = Conv1D(64, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.GRU(64, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

# Model Trainer with Nested CV and Threshold Optimization
class ModelTrainer:
    def __init__(self, model, X, y, groups):
        self.model = model
        self.X = X
        self.y = y
        self.groups = groups

    def tune_threshold(self, y_true, y_pred_proba):
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1, best_threshold = 0, 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        return best_threshold

    def nested_cv(self, n_outer_folds=5, n_inner_folds=5):
        outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds)
        results = []
        for train_idx, test_idx in outer_cv.split(self.X, self.y, self.groups):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            groups_train = self.groups[train_idx]

            # Inner CV for hyperparameter tuning
            def objective(trial):
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                model = self.model(input_shape=X_train.shape[1:])
                model.optimizer.learning_rate = lr
                model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_split=0.2, verbose=0)
                val_pred = model.predict(X_train)
                return accuracy_score(y_train, (val_pred > 0.5).astype(int))

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            best_params = study.best_params

            # Train with best params
            model = self.model(input_shape=X_train.shape[1:])
            model.optimizer.learning_rate = best_params['lr']
            model.fit(X_train, y_train, batch_size=best_params['batch_size'], epochs=50, verbose=0)
            y_pred_proba = model.predict(X_test)
            threshold = self.tune_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba > threshold).astype(int)

            # Aggregate predictions per subject
            unique_groups = np.unique(self.groups[test_idx])
            y_true_agg, y_pred_agg = [], []
            for group in unique_groups:
                idx = self.groups[test_idx] == group
                y_true_agg.append(y_test[idx][0])
                y_pred_agg.append(np.mean(y_pred_proba[idx]) > threshold)
            results.append({
                'accuracy': accuracy_score(y_true_agg, y_pred_agg),
                'precision': precision_recall_fscore_support(y_true_agg, y_pred_agg, average='binary')[0],
                'recall': precision_recall_fscore_support(y_true_agg, y_pred_agg, average='binary')[1],
                'f1': precision_recall_fscore_support(y_true_agg, y_pred_agg, average='binary')[2],
                'auc': roc_auc_score(y_true_agg, y_pred_agg)
            })
        return pd.DataFrame(results).mean()

# Main Pipeline
processor = EEGDataProcessor('ds004584-download')
file_paths = processor.gather_file_paths()
X, y, groups = processor.extract_windowed_data(file_paths)

# Split data
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train and evaluate 1D CNN
trainer = ModelTrainer(DeepLearningModels.build_1d_cnn, X_train_resampled, y_train_resampled, groups_train)
results = trainer.nested_cv()

# Interpretability with SHAP
model = DeepLearningModels.build_1d_cnn(input_shape=(2560, 64))
model.fit(X_train_resampled, y_train_resampled, epochs=50, verbose=0)
explainer = shap.DeepExplainer(model, X_train_resampled[:100])
shap_values = explainer.shap_values(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], feature_names=[f'Ch{i}_t{j}' for i in range(64) for j in range(2560)])

# Deployment with TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Visualize results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], y=[results['accuracy'], results['precision'], results['recall'], results['f1'], results['auc']])
plt.title('Model Performance')
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype(int)), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('performance.png')