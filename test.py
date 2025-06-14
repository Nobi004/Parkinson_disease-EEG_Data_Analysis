import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
from numba import jit
import warnings
import sys
import os

# Fix for Windows multiprocessing
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)

warnings.filterwarnings('ignore')

# Disable CUDA for multiprocessing compatibility on Windows
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class FastEEGPreprocessor:
    """Optimized EEG preprocessing pipeline with error handling"""
    
    def __init__(self, sampling_rate=512, channels=64, use_gpu=False, n_jobs=1):
        self.fs = sampling_rate
        self.channels = channels
        # Disable GPU for Windows multiprocessing compatibility
        self.use_gpu = False  # Force CPU to avoid multiprocessing issues
        # Limit jobs on Windows to avoid memory issues
        self.n_jobs = min(n_jobs if n_jobs > 0 else 1, 4)
        
        # Pre-compute filter coefficients
        self._precompute_filters()
    
    def _precompute_filters(self):
        """Pre-compute all filter coefficients"""
        nyquist = self.fs / 2
        
        # Bandpass filter (0.5-50 Hz)
        self.sos_bandpass = butter(4, [0.5/nyquist, 50/nyquist], btype='band', output='sos')
        
        # Notch filters
        self.sos_notch_50 = butter(4, [48/nyquist, 52/nyquist], btype='bandstop', output='sos')
        self.sos_notch_60 = butter(4, [58/nyquist, 62/nyquist], btype='bandstop', output='sos')
    
    def preprocess_batch(self, raw_data_batch, subject_ids, use_parallel=False):
        """
        Process multiple subjects (serial processing for Windows compatibility)
        """
        results = []
        for data, sid in zip(raw_data_batch, subject_ids):
            result = self._preprocess_single_safe(data, sid)
            results.append(result)
        
        return results
    
    def _preprocess_single_safe(self, raw_data, subject_id):
        """Safe preprocessing with comprehensive error handling"""
        try:
            # Validate input
            if not self._validate_data(raw_data):
                print(f"Invalid data for subject {subject_id}")
                return None
            
            # Convert to float32 and handle infinities
            data = self._sanitize_data(raw_data)
            
            # Processing steps
            data = self._safe_average_reference(data)
            data = self._safe_filter(data)
            data = self._safe_artifact_removal(data)
            epochs = self._safe_create_epochs(data)
            
            return epochs
            
        except Exception as e:
            print(f"Preprocessing failed for subject {subject_id}: {str(e)}")
            return None
    
    def _validate_data(self, data):
        """Validate input data"""
        if data is None or data.size == 0:
            return False
        if not np.isfinite(data).all():
            return False
        return True
    
    def _sanitize_data(self, data):
        """Clean and sanitize data"""
        data = data.astype(np.float32)
        
        # Replace infinities with large finite values
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values
        percentile_99 = np.percentile(np.abs(data[np.isfinite(data)]), 99)
        data = np.clip(data, -percentile_99 * 3, percentile_99 * 3)
        
        return data
    
    def _safe_average_reference(self, data):
        """Safe average referencing with validation"""
        try:
            # Check for bad channels
            channel_vars = np.var(data, axis=1)
            good_channels = np.isfinite(channel_vars) & (channel_vars > 0) & (channel_vars < np.percentile(channel_vars, 95))
            
            if np.sum(good_channels) < 2:
                # Not enough good channels, skip referencing
                return data
            
            # Compute average using only good channels
            avg_ref = np.mean(data[good_channels], axis=0)
            
            # Apply reference
            data = data - avg_ref
            
            return data
            
        except Exception as e:
            print(f"Average referencing failed: {e}")
            return data
    
    def _safe_filter(self, data):
        """Safe filtering with error handling"""
        try:
            # Check if data is valid for filtering
            if not np.isfinite(data).all():
                data = self._sanitize_data(data)
            
            # Apply bandpass filter
            filtered_data = sosfiltfilt(self.sos_bandpass, data, axis=1)
            
            # Check output
            if not np.isfinite(filtered_data).all():
                print("Warning: Filtering produced non-finite values, using original data")
                return data
            
            return filtered_data
            
        except Exception as e:
            print(f"Filtering failed: {e}")
            return data
    
    def _safe_artifact_removal(self, data):
        """Safe artifact removal"""
        try:
            # Compute robust statistics
            channel_vars = np.var(data, axis=1)
            finite_vars = channel_vars[np.isfinite(channel_vars)]
            
            if len(finite_vars) == 0:
                return data
            
            channel_median = np.median(finite_vars)
            
            # Identify bad channels
            bad_channels = ~np.isfinite(channel_vars) | (channel_vars > 5 * channel_median)
            
            # Simple interpolation
            if np.any(bad_channels) and np.sum(~bad_channels) > 0:
                good_channels = ~bad_channels
                for bad_idx in np.where(bad_channels)[0]:
                    # Find nearest good channel
                    distances = np.abs(np.arange(len(bad_channels)) - bad_idx)
                    distances[bad_channels] = np.inf
                    
                    if np.all(np.isinf(distances)):
                        # No good channels, zero out
                        data[bad_idx] = 0
                    else:
                        nearest_good = np.argmin(distances)
                        data[bad_idx] = data[nearest_good]
            
            # Remove high-amplitude artifacts
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                threshold = np.percentile(np.abs(finite_data), 99.5)
                data = np.clip(data, -threshold, threshold)
            
            return data
            
        except Exception as e:
            print(f"Artifact removal failed: {e}")
            return data
    
    def _safe_create_epochs(self, data, epoch_length=2.0, overlap=0.5):
        """Safe epoch creation"""
        try:
            epoch_samples = int(epoch_length * self.fs)
            step_size = int(epoch_samples * (1 - overlap))
            
            # Validate dimensions
            if data.shape[1] < epoch_samples:
                print("Warning: Data too short for requested epoch length")
                epoch_samples = data.shape[1]
                step_size = epoch_samples
            
            # Calculate number of epochs
            n_epochs = max(1, (data.shape[1] - epoch_samples) // step_size + 1)
            
            # Pre-allocate
            epochs = np.zeros((n_epochs, data.shape[0], epoch_samples), dtype=np.float32)
            
            # Create epochs
            for i in range(n_epochs):
                start = i * step_size
                end = start + epoch_samples
                
                if end > data.shape[1]:
                    # Pad last epoch if necessary
                    padded_epoch = np.zeros((data.shape[0], epoch_samples))
                    actual_length = data.shape[1] - start
                    padded_epoch[:, :actual_length] = data[:, start:]
                    epochs[i] = padded_epoch
                else:
                    epochs[i] = data[:, start:end]
            
            return epochs
            
        except Exception as e:
            print(f"Epoch creation failed: {e}")
            # Return at least one epoch
            return data[:, :min(data.shape[1], epoch_samples)].reshape(1, data.shape[0], -1)


class FastFeatureExtractor:
    """Safe feature extraction with error handling"""
    
    def __init__(self, sampling_rate=512, use_gpu=False, n_jobs=1):
        self.fs = sampling_rate
        self.use_gpu = False  # Force CPU
        self.n_jobs = 1  # Serial processing for safety
        
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def extract_features_batch(self, epochs_batch, use_parallel=False):
        """Extract features with error handling"""
        if epochs_batch is None or len(epochs_batch) == 0:
            return np.array([])
        
        features = []
        for epoch in epochs_batch:
            try:
                epoch_features = self._extract_features_safe(epoch)
                features.append(epoch_features)
            except Exception as e:
                print(f"Feature extraction failed for epoch: {e}")
                # Return zero features for failed epoch
                features.append(np.zeros(self._get_feature_dim()))
        
        return np.array(features, dtype=np.float32)
    
    def _get_feature_dim(self):
        """Get expected feature dimension"""
        # 5 bands * 2 (absolute + relative) * n_channels + statistical features
        n_channels = 64  # Default
        return 5 * 2 * n_channels + 4 * n_channels
    
    def _extract_features_safe(self, epoch):
        """Safe feature extraction"""
        n_channels, n_samples = epoch.shape
        features = []
        
        # Validate epoch
        if not np.isfinite(epoch).all():
            epoch = np.nan_to_num(epoch, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            # Compute FFT
            fft_data = np.fft.rfft(epoch, axis=1)
            fft_magnitude = np.abs(fft_data)
            freqs = np.fft.rfftfreq(n_samples, 1/self.fs)
            
            # Band powers
            for band_name, (low, high) in self.freq_bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_powers = np.sum(fft_magnitude[:, band_mask], axis=1)
                else:
                    band_powers = np.zeros(n_channels)
                
                # Handle NaN/Inf
                band_powers = np.nan_to_num(band_powers, nan=0.0, posinf=0.0, neginf=0.0)
                features.extend(band_powers)
            
            # Relative powers
            total_powers = np.sum(fft_magnitude, axis=1)
            total_powers[total_powers == 0] = 1e-10  # Avoid division by zero
            
            for band_name, (low, high) in self.freq_bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    relative_powers = np.sum(fft_magnitude[:, band_mask], axis=1) / total_powers
                else:
                    relative_powers = np.zeros(n_channels)
                
                relative_powers = np.nan_to_num(relative_powers, nan=0.0, posinf=1.0, neginf=0.0)
                features.extend(relative_powers)
            
            # Statistical features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features.extend(np.nan_to_num(np.mean(epoch, axis=1), nan=0.0))
                features.extend(np.nan_to_num(np.std(epoch, axis=1), nan=0.0))
                features.extend(np.nan_to_num(np.percentile(epoch, 25, axis=1), nan=0.0))
                features.extend(np.nan_to_num(np.percentile(epoch, 75, axis=1), nan=0.0))
            
        except Exception as e:
            print(f"Feature computation error: {e}")
            # Return zeros if feature extraction fails
            return np.zeros(self._get_feature_dim())
        
        return np.array(features, dtype=np.float32)


class OptimizedPipeline:
    """Stable pipeline with comprehensive error handling"""
    
    def __init__(self, use_gpu=False, n_jobs=1, cache_features=True):
        # Force safe settings for Windows
        self.preprocessor = FastEEGPreprocessor(use_gpu=False, n_jobs=1)
        self.feature_extractor = FastFeatureExtractor(use_gpu=False, n_jobs=1)
        self.cache_features = cache_features
        self.feature_cache = {}
    
    def process_dataset(self, raw_data_list, subject_ids, batch_size=8):
        """Process dataset with error recovery"""
        all_features = []
        successful_subjects = []
        
        n_subjects = len(raw_data_list)
        
        for i in range(0, n_subjects, batch_size):
            batch_data = raw_data_list[i:i+batch_size]
            batch_ids = subject_ids[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(n_subjects + batch_size - 1)//batch_size}")
            
            # Check cache
            if self.cache_features:
                cached_features, uncached_indices = self._check_cache(batch_ids)
                
                if len(uncached_indices) == 0:
                    all_features.extend(cached_features)
                    successful_subjects.extend(batch_ids)
                    continue
                
                batch_data = [batch_data[idx] for idx in uncached_indices]
                batch_ids = [batch_ids[idx] for idx in uncached_indices]
            
            # Preprocess batch
            preprocessed_batch = self.preprocessor.preprocess_batch(batch_data, batch_ids)
            
            # Extract features
            for j, (epochs, subject_id) in enumerate(zip(preprocessed_batch, batch_ids)):
                if epochs is not None:
                    try:
                        features = self.feature_extractor.extract_features_batch(epochs)
                        
                        if features.size > 0:
                            if self.cache_features:
                                self.feature_cache[subject_id] = features
                            
                            all_features.append(features)
                            successful_subjects.append(subject_id)
                    except Exception as e:
                        print(f"Feature extraction failed for {subject_id}: {e}")
        
        if all_features:
            return np.vstack(all_features), successful_subjects
        else:
            return np.array([]), []
    
    def _check_cache(self, subject_ids):
        """Check cache safely"""
        cached_features = []
        uncached_indices = []
        
        for i, subject_id in enumerate(subject_ids):
            if subject_id in self.feature_cache:
                cached_features.extend(self.feature_cache[subject_id])
            else:
                uncached_indices.append(i)
        
        return cached_features, uncached_indices


def benchmark_preprocessing():
    """Safe benchmarking"""
    import time
    
    # Generate synthetic data
    n_subjects = 149
    n_channels = 64
    n_samples = 1024
    
    print("Generating synthetic data...")
    np.random.seed(42)
    
    # Create more realistic synthetic data
    raw_data_list = []
    for i in range(n_subjects):
        # Simulate realistic EEG with occasional artifacts
        data = np.random.randn(n_channels, n_samples).astype(np.float32) * 50  # μV scale
        
        # Add some channels with artifacts
        if i % 20 == 0:  # 5% of subjects have artifacts
            artifact_channels = np.random.choice(n_channels, size=5, replace=False)
            data[artifact_channels] *= 10  # Large artifacts
        
        raw_data_list.append(data)
    
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Test safe configuration
    configs = [
        {"use_gpu": False, "n_jobs": 1, "name": "Safe Single CPU"},
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        pipeline = OptimizedPipeline(
            use_gpu=config['use_gpu'],
            n_jobs=config['n_jobs'],
            cache_features=False
        )
        
        start_time = time.time()
        features, successful = pipeline.process_dataset(raw_data_list, subject_ids, batch_size=8)
        end_time = time.time()
        
        processing_time = end_time - start_time
        if len(successful) > 0:
            subjects_per_second = len(successful) / processing_time
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Successful subjects: {len(successful)}/{n_subjects}")
            print(f"Subjects per second: {subjects_per_second:.2f}")
            print(f"Features shape: {features.shape}")
        else:
            print("No subjects processed successfully")


def main():
    """Safe main execution"""
    # Run benchmark
    benchmark_preprocessing()
    
    # Example usage
    print("\n" + "="*60)
    print("Example usage of optimized pipeline:")
    
    # Initialize pipeline
    pipeline = OptimizedPipeline(use_gpu=False, n_jobs=1, cache_features=True)
    
    # Simulate processing
    n_subjects = 10
    raw_data = [np.random.randn(64, 1024).astype(np.float32) * 50 for _ in range(n_subjects)]
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Process data
    features, successful = pipeline.process_dataset(raw_data, subject_ids, batch_size=4)
    print(f"Processed features shape: {features.shape}")
    print(f"Successful subjects: {len(successful)}/{n_subjects}")


if __name__ == "__main__":
    # Ensure proper execution on Windows
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    
    main()