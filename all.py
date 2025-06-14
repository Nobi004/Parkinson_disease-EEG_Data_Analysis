import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import mne
from scipy.signal import butter, sosfilt, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
from numba import jit, cuda
import cupy as cp  # GPU acceleration for NumPy operations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class FastEEGPreprocessor:
    """Highly optimized EEG preprocessing pipeline with GPU acceleration"""
    
    def __init__(self, sampling_rate=512, channels=64, use_gpu=True, n_jobs=-1):
        self.fs = sampling_rate
        self.channels = channels
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        # Pre-compute filter coefficients (one-time computation)
        self._precompute_filters()
        
        # Initialize GPU arrays if available
        if self.use_gpu:
            self._init_gpu_arrays()
    
    def _precompute_filters(self):
        """Pre-compute all filter coefficients to avoid repeated computation"""
        nyquist = self.fs / 2
        
        # Bandpass filter (0.5-50 Hz)
        self.sos_bandpass = butter(4, [0.5/nyquist, 50/nyquist], btype='band', output='sos')
        
        # Notch filters for power line noise
        self.sos_notch_50 = butter(4, [48/nyquist, 52/nyquist], btype='bandstop', output='sos')
        self.sos_notch_60 = butter(4, [58/nyquist, 62/nyquist], btype='bandstop', output='sos')
        
        # Sub-band filters for feature extraction
        self.band_filters = {}
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for band_name, (low, high) in freq_bands.items():
            self.band_filters[band_name] = butter(4, [low/nyquist, high/nyquist], 
                                                 btype='band', output='sos')
    
    def _init_gpu_arrays(self):
        """Initialize GPU arrays for faster computation"""
        # Pre-allocate GPU memory for common array sizes
        self.gpu_temp_arrays = {
            'filter': cp.zeros((self.channels, self.fs * 2), dtype=cp.float32),
            'reference': cp.zeros(self.fs * 2, dtype=cp.float32),
            'epochs': cp.zeros((10, self.channels, self.fs * 2), dtype=cp.float32)
        }
    
    def preprocess_batch(self, raw_data_batch, subject_ids, use_parallel=True):
        """
        Process multiple subjects in parallel
        
        Args:
            raw_data_batch: List of EEG data arrays (n_subjects x channels x time_points)
            subject_ids: List of subject identifiers
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of preprocessed EEG epochs for each subject
        """
        if use_parallel:
            # Use multiprocessing for CPU-bound tasks
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(
                    self._preprocess_single_fast,
                    raw_data_batch,
                    subject_ids
                ))
        else:
            results = [self._preprocess_single_fast(data, sid) 
                      for data, sid in zip(raw_data_batch, subject_ids)]
        
        return results
    
    def _preprocess_single_fast(self, raw_data, subject_id):
        """
        Optimized single-subject preprocessing
        """
        try:
            # Convert to float32 for faster computation
            data = raw_data.astype(np.float32)
            
            # GPU acceleration if available
            if self.use_gpu and cp:
                return self._preprocess_gpu(data, subject_id)
            else:
                return self._preprocess_cpu(data, subject_id)
                
        except Exception as e:
            print(f"Preprocessing failed for subject {subject_id}: {str(e)}")
            return None
    
    def _preprocess_gpu(self, data, subject_id):
        """GPU-accelerated preprocessing using CuPy"""
        # Transfer to GPU
        gpu_data = cp.asarray(data, dtype=cp.float32)
        
        # Step 1: Fast average referencing on GPU
        gpu_data = self._gpu_average_reference(gpu_data)
        
        # Step 2: Apply filters on GPU
        gpu_data = self._gpu_filter(gpu_data)
        
        # Step 3: Simple artifact removal
        gpu_data = self._gpu_artifact_removal(gpu_data)
        
        # Step 4: Create epochs
        epochs = self._gpu_create_epochs(gpu_data)
        
        # Transfer back to CPU
        return cp.asnumpy(epochs)
    
    def _preprocess_cpu(self, data, subject_id):
        """CPU preprocessing with numba optimization"""
        # Step 1: Fast average referencing
        data = self._fast_average_reference(data)
        
        # Step 2: Vectorized filtering
        data = self._vectorized_filter(data)
        
        # Step 3: Simple artifact removal
        data = self._simple_artifact_removal(data)
        
        # Step 4: Create epochs efficiently
        epochs = self._fast_create_epochs(data)
        
        return epochs
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_average_reference(data):
        """Numba-optimized average referencing"""
        n_channels, n_samples = data.shape
        avg_ref = np.zeros(n_samples, dtype=np.float32)
        
        # Compute average reference
        for i in range(n_samples):
            for ch in range(n_channels):
                avg_ref[i] += data[ch, i]
            avg_ref[i] /= n_channels
        
        # Apply reference
        for ch in range(n_channels):
            for i in range(n_samples):
                data[ch, i] -= avg_ref[i]
        
        return data
    
    def _vectorized_filter(self, data):
        """Vectorized filtering for all channels simultaneously"""
        # Apply filters in place to save memory
        data = sosfiltfilt(self.sos_bandpass, data, axis=1)
        
        # Apply notch filter if needed (detect power line frequency)
        if self._detect_powerline_noise(data):
            data = sosfiltfilt(self.sos_notch_50, data, axis=1)
        
        return data
    
    def _simple_artifact_removal(self, data):
        """Fast artifact removal using statistical thresholds"""
        # Compute channel statistics
        channel_vars = np.var(data, axis=1)
        channel_median = np.median(channel_vars)
        
        # Identify bad channels (>5x median variance)
        bad_channels = channel_vars > 5 * channel_median
        
        # Interpolate bad channels using nearest neighbors
        if np.any(bad_channels):
            good_channels = ~bad_channels
            for bad_idx in np.where(bad_channels)[0]:
                # Find nearest good channels
                distances = np.abs(np.arange(len(bad_channels)) - bad_idx)
                distances[bad_channels] = np.inf
                nearest_good = np.argmin(distances)
                data[bad_idx] = data[nearest_good]
        
        # Remove high-amplitude artifacts
        threshold = np.percentile(np.abs(data), 99.5)
        data = np.clip(data, -threshold, threshold)
        
        return data
    
    def _fast_create_epochs(self, data, epoch_length=2.0, overlap=0.5):
        """Efficient epoch creation using array views"""
        epoch_samples = int(epoch_length * self.fs)
        step_size = int(epoch_samples * (1 - overlap))
        
        # Calculate number of epochs
        n_epochs = (data.shape[1] - epoch_samples) // step_size + 1
        
        # Pre-allocate epoch array
        epochs = np.zeros((n_epochs, data.shape[0], epoch_samples), dtype=np.float32)
        
        # Use memory views for efficient copying
        for i in range(n_epochs):
            start = i * step_size
            epochs[i] = data[:, start:start + epoch_samples]
        
        return epochs
    
    def _detect_powerline_noise(self, data):
        """Quick detection of power line noise"""
        # Use only first few channels for speed
        test_channels = min(5, data.shape[0])
        
        # Quick FFT on subset
        freqs = np.fft.rfftfreq(data.shape[1], 1/self.fs)
        
        for ch in range(test_channels):
            fft = np.abs(np.fft.rfft(data[ch]))
            
            # Check for peaks at 50Hz or 60Hz
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            
            if fft[idx_50] > 5 * np.median(fft) or fft[idx_60] > 5 * np.median(fft):
                return True
        
        return False
    
    # GPU-accelerated functions
    def _gpu_average_reference(self, gpu_data):
        """GPU average referencing using CuPy"""
        avg_ref = cp.mean(gpu_data, axis=0)
        gpu_data -= avg_ref
        return gpu_data
    
    def _gpu_filter(self, gpu_data):
        """GPU filtering using CuPy"""
        # CuPy doesn't have sosfiltfilt, so we transfer to CPU for filtering
        # In practice, you might want to implement GPU filtering using CUDA kernels
        cpu_data = cp.asnumpy(gpu_data)
        cpu_data = sosfiltfilt(self.sos_bandpass, cpu_data, axis=1)
        return cp.asarray(cpu_data)
    
    def _gpu_artifact_removal(self, gpu_data):
        """GPU artifact removal"""
        # Compute statistics on GPU
        channel_vars = cp.var(gpu_data, axis=1)
        threshold = cp.percentile(cp.abs(gpu_data), 99.5)
        
        # Clip values
        gpu_data = cp.clip(gpu_data, -threshold, threshold)
        
        return gpu_data
    
    def _gpu_create_epochs(self, gpu_data):
        """GPU epoch creation"""
        epoch_length = 2.0
        overlap = 0.5
        epoch_samples = int(epoch_length * self.fs)
        step_size = int(epoch_samples * (1 - overlap))
        
        n_epochs = (gpu_data.shape[1] - epoch_samples) // step_size + 1
        
        # Pre-allocate on GPU
        epochs = cp.zeros((n_epochs, gpu_data.shape[0], epoch_samples), dtype=cp.float32)
        
        for i in range(n_epochs):
            start = i * step_size
            epochs[i] = gpu_data[:, start:start + epoch_samples]
        
        return epochs


class FastFeatureExtractor:
    """Optimized feature extraction with parallel processing"""
    
    def __init__(self, sampling_rate=512, use_gpu=True, n_jobs=-1):
        self.fs = sampling_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        
        # Pre-compute frequency bins
        self._precompute_freq_bins()
    
    def _precompute_freq_bins(self):
        """Pre-compute frequency bins for faster lookup"""
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Pre-compute FFT frequencies for common epoch lengths
        common_lengths = [512, 1024, 2048]  # Common epoch sample counts
        self.fft_freqs = {}
        for length in common_lengths:
            self.fft_freqs[length] = np.fft.rfftfreq(length, 1/self.fs)
    
    def extract_features_batch(self, epochs_batch, use_parallel=True):
        """
        Extract features from multiple epochs in parallel
        
        Args:
            epochs_batch: Array of epochs (n_epochs, n_channels, n_samples)
            use_parallel: Whether to use parallel processing
            
        Returns:
            Feature matrix (n_epochs, n_features)
        """
        if use_parallel:
            # Split epochs for parallel processing
            n_epochs = len(epochs_batch)
            chunk_size = max(1, n_epochs // self.n_jobs)
            
            # Use ThreadPoolExecutor for I/O-bound operations
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i in range(0, n_epochs, chunk_size):
                    chunk = epochs_batch[i:i+chunk_size]
                    futures.append(executor.submit(self._extract_features_chunk, chunk))
                
                # Collect results
                features = []
                for future in futures:
                    features.extend(future.result())
        else:
            features = self._extract_features_chunk(epochs_batch)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features_chunk(self, epochs_chunk):
        """Extract features from a chunk of epochs"""
        features = []
        
        for epoch in epochs_chunk:
            if self.use_gpu and torch.cuda.is_available():
                epoch_features = self._extract_features_gpu(epoch)
            else:
                epoch_features = self._extract_features_vectorized(epoch)
            
            features.append(epoch_features)
        
        return features
    
    def _extract_features_vectorized(self, epoch):
        """Vectorized feature extraction for single epoch"""
        n_channels, n_samples = epoch.shape
        features = []
        
        # Compute FFT for all channels at once
        fft_data = np.fft.rfft(epoch, axis=1)
        fft_magnitude = np.abs(fft_data)
        
        # Get frequency array
        if n_samples in self.fft_freqs:
            freqs = self.fft_freqs[n_samples]
        else:
            freqs = np.fft.rfftfreq(n_samples, 1/self.fs)
        
        # Extract band powers for all channels vectorized
        for band_name, (low, high) in self.freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_powers = np.sum(fft_magnitude[:, band_mask], axis=1)
            features.extend(band_powers)
        
        # Compute relative powers
        total_powers = np.sum(fft_magnitude, axis=1)
        for band_name, (low, high) in self.freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            relative_powers = np.sum(fft_magnitude[:, band_mask], axis=1) / (total_powers + 1e-10)
            features.extend(relative_powers)
        
        # Statistical features (vectorized)
        features.extend(np.mean(epoch, axis=1))
        features.extend(np.std(epoch, axis=1))
        features.extend(np.percentile(epoch, 25, axis=1))
        features.extend(np.percentile(epoch, 75, axis=1))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_features_gpu(self, epoch):
        """GPU-accelerated feature extraction"""
        # Convert to GPU tensor
        epoch_gpu = torch.from_numpy(epoch).float().cuda()
        
        # Compute FFT on GPU
        fft_data = torch.fft.rfft(epoch_gpu, dim=1)
        fft_magnitude = torch.abs(fft_data)
        
        # Extract features on GPU
        features = []
        
        # Band powers
        n_samples = epoch.shape[1]
        freqs = torch.fft.rfftfreq(n_samples, 1/self.fs).cuda()
        
        for band_name, (low, high) in self.freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_powers = torch.sum(fft_magnitude[:, band_mask], dim=1)
            features.append(band_powers.cpu().numpy())
        
        # Statistical features
        features.append(torch.mean(epoch_gpu, dim=1).cpu().numpy())
        features.append(torch.std(epoch_gpu, dim=1).cpu().numpy())
        
        return np.concatenate(features)


class OptimizedPipeline:
    """Complete optimized preprocessing and feature extraction pipeline"""
    
    def __init__(self, use_gpu=True, n_jobs=-1, cache_features=True):
        self.preprocessor = FastEEGPreprocessor(use_gpu=use_gpu, n_jobs=n_jobs)
        self.feature_extractor = FastFeatureExtractor(use_gpu=use_gpu, n_jobs=n_jobs)
        self.cache_features = cache_features
        self.feature_cache = {}
    
    def process_dataset(self, raw_data_list, subject_ids, batch_size=8):
        """
        Process entire dataset with optimizations
        
        Args:
            raw_data_list: List of raw EEG data arrays
            subject_ids: List of subject identifiers
            batch_size: Number of subjects to process in parallel
            
        Returns:
            Processed features and labels
        """
        all_features = []
        
        # Process in batches
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
                    continue
                
                # Process only uncached subjects
                batch_data = [batch_data[idx] for idx in uncached_indices]
                batch_ids = [batch_ids[idx] for idx in uncached_indices]
            
            # Preprocess batch
            preprocessed_batch = self.preprocessor.preprocess_batch(batch_data, batch_ids)
            
            # Extract features for each subject
            for j, (epochs, subject_id) in enumerate(zip(preprocessed_batch, batch_ids)):
                if epochs is not None:
                    features = self.feature_extractor.extract_features_batch(epochs)
                    
                    if self.cache_features:
                        self.feature_cache[subject_id] = features
                    
                    all_features.append(features)
        
        return np.vstack(all_features) if all_features else np.array([])
    
    def _check_cache(self, subject_ids):
        """Check which subjects are already in cache"""
        cached_features = []
        uncached_indices = []
        
        for i, subject_id in enumerate(subject_ids):
            if subject_id in self.feature_cache:
                cached_features.extend(self.feature_cache[subject_id])
            else:
                uncached_indices.append(i)
        
        return cached_features, uncached_indices
    
    def save_cache(self, filepath):
        """Save feature cache to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.feature_cache, f)
    
    def load_cache(self, filepath):
        """Load feature cache from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            self.feature_cache = pickle.load(f)


# Example usage with benchmarking
def benchmark_preprocessing():
    """Benchmark the optimized preprocessing pipeline"""
    import time
    
    # Simulate data
    n_subjects = 149
    n_channels = 64
    n_samples = 1024
    
    print("Generating synthetic data...")
    raw_data_list = [np.random.randn(n_channels, n_samples).astype(np.float32) 
                     for _ in range(n_subjects)]
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Test different configurations
    configs = [
        {"use_gpu": False, "n_jobs": 1, "name": "Single CPU"},
        {"use_gpu": False, "n_jobs": -1, "name": "Multi CPU"},
        {"use_gpu": True, "n_jobs": -1, "name": "GPU + Multi CPU"},
    ]
    
    for config in configs:
        print(f"\n Testing {config['name']} configuration...")
        pipeline = OptimizedPipeline(
            use_gpu=config['use_gpu'],
            n_jobs=config['n_jobs'],
            cache_features=False
        )
        
        start_time = time.time()
        features = pipeline.process_dataset(raw_data_list, subject_ids, batch_size=8)
        end_time = time.time()
        
        processing_time = end_time - start_time
        subjects_per_second = n_subjects / processing_time
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Subjects per second: {subjects_per_second:.2f}")
        print(f"Features shape: {features.shape}")


# Memory-efficient data loader
class EEGDataLoader:
    """Memory-efficient data loader with prefetching"""
    
    def __init__(self, data_path, batch_size=8, prefetch_size=2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.prefetch_queue = []
        self.prefetch_thread = None
    
    def __iter__(self):
        """Iterate through data with prefetching"""
        # Start prefetching thread
        from threading import Thread
        import queue
        
        self.data_queue = queue.Queue(maxsize=self.prefetch_size)
        self.prefetch_thread = Thread(target=self._prefetch_worker)
        self.prefetch_thread.start()
        
        # Yield batches
        while True:
            try:
                batch = self.data_queue.get(timeout=30)
                if batch is None:  # End signal
                    break
                yield batch
            except queue.Empty:
                print("Warning: Data loading timeout")
                break
        
        # Clean up
        self.prefetch_thread.join()
    
    def _prefetch_worker(self):
        """Worker thread for prefetching data"""
        # Implementation depends on your data format
        # This is a placeholder
        pass


if __name__ == "__main__":
    # Run benchmark
    benchmark_preprocessing()
    
    # Example of using the optimized pipeline
    print("\n" + "="*60)
    print("Example usage of optimized pipeline:")
    
    # Initialize pipeline
    pipeline = OptimizedPipeline(use_gpu=True, n_jobs=-1, cache_features=True)
    
    # Simulate processing
    n_subjects = 10
    raw_data = [np.random.randn(64, 1024).astype(np.float32) for _ in range(n_subjects)]
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Process data
    features = pipeline.process_dataset(raw_data, subject_ids, batch_size=4)
    print(f"Processed features shape: {features.shape}")
    
    # Save cache for future use
    pipeline.save_cache("feature_cache.pkl")
    print("Feature cache saved!")