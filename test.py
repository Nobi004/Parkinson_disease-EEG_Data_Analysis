import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class WindowsOptimizedEEGProcessor:
    """
    Fast EEG processing optimized for Windows without multiprocessing issues
    """
    
    def __init__(self, sampling_rate=512, channels=64):
        self.fs = sampling_rate
        self.channels = channels
        
        # Pre-compute filters once
        self._init_filters()
        
        # Pre-allocate arrays for better performance
        self._init_buffers()
    
    def _init_filters(self):
        """Initialize all filters at once"""
        nyquist = self.fs / 2
        
        # Main filters
        self.sos_bandpass = butter(4, [0.5/nyquist, 50/nyquist], btype='band', output='sos')
        
        # Band-specific filters for features
        self.band_filters = {}
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
                'beta': (13, 30), 'gamma': (30, 50)}
        
        for name, (low, high) in bands.items():
            self.band_filters[name] = butter(3, [low/nyquist, high/nyquist], 
                                           btype='band', output='sos')
    
    def _init_buffers(self):
        """Pre-allocate buffers for better memory performance"""
        # Typical epoch size
        self.epoch_length = 2.0
        self.epoch_samples = int(self.epoch_length * self.fs)
        
        # Pre-allocate work buffers
        self.filter_buffer = np.zeros((self.channels, self.epoch_samples * 2))
        self.fft_buffer = np.zeros((self.channels, self.epoch_samples // 2 + 1), dtype=complex)
    
    def process_batch_fast(self, data_list, subject_ids):
        """
        Process multiple subjects efficiently without multiprocessing
        """
        all_features = []
        
        for data, subject_id in zip(data_list, subject_ids):
            try:
                # Quick validation
                if not self._quick_validate(data):
                    print(f"Skipping invalid data for {subject_id}")
                    continue
                
                # Fast preprocessing
                clean_data = self._fast_preprocess(data)
                
                # Fast feature extraction
                features = self._fast_features(clean_data)
                
                all_features.extend(features)
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                continue
        
        return np.array(all_features) if all_features else np.array([])
    
    def _quick_validate(self, data):
        """Quick data validation"""
        if data is None or data.size == 0:
            return False
        
        # Check for NaN/Inf
        if not np.isfinite(data).all():
            return False
        
        # Check reasonable range (μV)
        if np.abs(data).max() > 1e6:  # 1 mV is huge for EEG
            return False
        
        return True
    
    def _fast_preprocess(self, data):
        """Streamlined preprocessing"""
        # Ensure float32 for speed
        data = data.astype(np.float32)
        
        # 1. Remove bad channels (simple threshold)
        channel_powers = np.sqrt(np.mean(data**2, axis=1))
        median_power = np.median(channel_powers)
        good_channels = (channel_powers > 0.1 * median_power) & \
                       (channel_powers < 10 * median_power)
        
        # 2. Average reference using good channels only
        if np.sum(good_channels) > 5:
            ref = np.mean(data[good_channels], axis=0)
            data = data - ref
        
        # 3. Bandpass filter (vectorized)
        data = sosfiltfilt(self.sos_bandpass, data, axis=1)
        
        # 4. Simple artifact rejection
        threshold = np.percentile(np.abs(data), 99)
        data = np.clip(data, -threshold, threshold)
        
        return data
    
    def _fast_features(self, data):
        """Fast feature extraction using vectorized operations"""
        features_list = []
        
        # Create epochs efficiently
        n_samples = data.shape[1]
        epoch_samples = min(self.epoch_samples, n_samples)
        step = epoch_samples // 2  # 50% overlap
        
        for start in range(0, n_samples - epoch_samples + 1, step):
            epoch = data[:, start:start + epoch_samples]
            
            # Compute features
            features = self._compute_epoch_features(epoch)
            features_list.append(features)
        
        return features_list
    
    def _compute_epoch_features(self, epoch):
        """Compute features for single epoch - highly optimized"""
        features = []
        
        # 1. FFT-based features (all channels at once)
        fft_data = np.fft.rfft(epoch, axis=1)
        fft_mag = np.abs(fft_data)
        freqs = np.fft.rfftfreq(epoch.shape[1], 1/self.fs)
        
        # Band powers (vectorized)
        for band, (low, high) in [('delta', (0.5, 4)), ('theta', (4, 8)), 
                                  ('alpha', (8, 13)), ('beta', (13, 30)), 
                                  ('gamma', (30, 50))]:
            band_idx = (freqs >= low) & (freqs <= high)
            band_power = np.sum(fft_mag[:, band_idx], axis=1)
            features.extend(band_power)
        
        # 2. Time-domain features (vectorized)
        features.extend(np.mean(epoch, axis=1))      # Mean
        features.extend(np.std(epoch, axis=1))       # Std
        features.extend(np.max(epoch, axis=1))       # Max
        features.extend(np.min(epoch, axis=1))       # Min
        
        # 3. Statistical features
        features.extend(np.percentile(epoch, 25, axis=1))  # Q1
        features.extend(np.percentile(epoch, 75, axis=1))  # Q3
        
        return np.array(features, dtype=np.float32)


class GPUAcceleratedProcessor:
    """
    GPU-accelerated processing using PyTorch (when multiprocessing not needed)
    """
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Move filters to GPU
        self.fs = 512
        self._init_gpu_filters()
    
    def _init_gpu_filters(self):
        """Initialize GPU-based filters"""
        # Create filter banks on GPU
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def process_batch_gpu(self, data_batch):
        """Process batch on GPU"""
        # Convert to tensor
        data_tensor = torch.from_numpy(np.array(data_batch)).float().to(self.device)
        
        # Batch processing on GPU
        with torch.no_grad():
            # Preprocessing
            clean_data = self._preprocess_gpu(data_tensor)
            
            # Feature extraction
            features = self._extract_features_gpu(clean_data)
        
        return features.cpu().numpy()
    
    def _preprocess_gpu(self, data):
        """GPU preprocessing"""
        batch_size, channels, samples = data.shape
        
        # Average reference
        ref = torch.mean(data, dim=1, keepdim=True)
        data = data - ref
        
        # Simple high-pass filter (remove DC)
        data = data - torch.mean(data, dim=2, keepdim=True)
        
        # Artifact clipping
        threshold = torch.quantile(torch.abs(data), 0.99)
        data = torch.clamp(data, -threshold, threshold)
        
        return data
    
    def _extract_features_gpu(self, data):
        """GPU feature extraction"""
        batch_size, channels, samples = data.shape
        
        # FFT on GPU
        fft_data = torch.fft.rfft(data, dim=2)
        fft_mag = torch.abs(fft_data)
        
        # Compute band powers
        features = []
        freqs = torch.fft.rfftfreq(samples, 1/self.fs).to(self.device)
        
        for band, (low, high) in self.freq_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = torch.sum(fft_mag[:, :, band_mask], dim=2)
            features.append(band_power)
        
        # Statistical features
        features.append(torch.mean(data, dim=2))
        features.append(torch.std(data, dim=2))
        features.append(torch.quantile(data, 0.25, dim=2))
        features.append(torch.quantile(data, 0.75, dim=2))
        
        # Concatenate all features
        all_features = torch.cat(features, dim=1)
        
        return all_features


def optimize_for_your_system():
    """
    Recommendations for your specific setup
    """
    print("=== Optimization Recommendations for Your System ===\n")
    
    print("1. INCREASE WINDOWS PAGE FILE:")
    print("   - Go to: System Properties > Advanced > Performance Settings")
    print("   - Advanced tab > Virtual Memory > Change")
    print("   - Set to: System managed size OR")
    print("   - Custom size: Initial 16384 MB, Maximum 32768 MB")
    print("   - Restart your computer\n")
    
    print("2. USE OPTIMIZED BATCH SIZES:")
    print("   - For RTX 4060 (8GB): Use batch_size=4")
    print("   - Process subjects sequentially, not in parallel")
    print("   - Clear GPU cache frequently\n")
    
    print("3. MEMORY MANAGEMENT:")
    print("   ```python")
    print("   import gc")
    print("   import torch")
    print("   ")
    print("   # After each batch:")
    print("   gc.collect()")
    print("   torch.cuda.empty_cache()")
    print("   ```\n")
    
    print("4. USE THE WINDOWS-OPTIMIZED PROCESSOR:")
    print("   ```python")
    print("   processor = WindowsOptimizedEEGProcessor()")
    print("   features = processor.process_batch_fast(data_list, subject_ids)")
    print("   ```")


# Complete working example
def run_optimized_pipeline():
    """
    Complete working pipeline for Windows
    """
    import time
    import gc
    
    # Generate test data
    n_subjects = 149
    n_channels = 64
    n_samples = 1024
    
    print("Initializing optimized processor...")
    processor = WindowsOptimizedEEGProcessor(sampling_rate=512, channels=n_channels)
    
    print("Generating synthetic data...")
    np.random.seed(42)
    data_list = []
    for i in range(n_subjects):
        # Realistic EEG data (microvolts)
        data = np.random.randn(n_channels, n_samples).astype(np.float32) * 50
        data_list.append(data)
    
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Process in small batches to avoid memory issues
    batch_size = 10
    all_features = []
    
    start_time = time.time()
    
    for i in range(0, n_subjects, batch_size):
        batch_data = data_list[i:i+batch_size]
        batch_ids = subject_ids[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(n_subjects + batch_size - 1)//batch_size}")
        
        # Process batch
        features = processor.process_batch_fast(batch_data, batch_ids)
        
        if features.size > 0:
            all_features.append(features)
        
        # Clean up memory
        gc.collect()
    
    # Combine results
    if all_features:
        final_features = np.vstack(all_features)
        processing_time = time.time() - start_time
        
        print(f"\nProcessing completed!")
        print(f"Total time: {processing_time:.2f} seconds")
        print(f"Subjects per second: {n_subjects/processing_time:.2f}")
        print(f"Final features shape: {final_features.shape}")
    else:
        print("No features extracted")


# GPU version (single-threaded, no multiprocessing)
def run_gpu_pipeline():
    """
    GPU-accelerated pipeline without multiprocessing
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU version instead.")
        return run_optimized_pipeline()
    
    import time
    
    # Initialize GPU processor
    gpu_processor = GPUAcceleratedProcessor(device='cuda')
    
    # Generate test data
    n_subjects = 149
    n_channels = 64
    n_samples = 1024
    
    print("Generating synthetic data...")
    data_list = []
    for i in range(n_subjects):
        data = np.random.randn(n_channels, n_samples).astype(np.float32) * 50
        data_list.append(data)
    
    # Process on GPU in batches
    batch_size = 16  # Larger batches OK for GPU
    all_features = []
    
    start_time = time.time()
    
    for i in range(0, n_subjects, batch_size):
        batch = data_list[i:i+batch_size]
        
        print(f"GPU Processing batch {i//batch_size + 1}")
        
        # Process on GPU
        features = gpu_processor.process_batch_gpu(batch)
        all_features.append(features)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Combine results
    final_features = np.vstack(all_features)
    processing_time = time.time() - start_time
    
    print(f"\nGPU Processing completed!")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Subjects per second: {n_subjects/processing_time:.2f}")
    print(f"Final features shape: {final_features.shape}")


# Main execution
if __name__ == "__main__":
    import sys
    
    print("EEG Processing Pipeline - Windows Optimized Version\n")
    
    # Show optimization tips
    optimize_for_your_system()
    
    print("\n" + "="*60 + "\n")
    
    # Check available resources
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "="*60 + "\n")
    
    # Run CPU version (most stable)
    print("Running CPU-optimized version...")
    run_optimized_pipeline()
    
    print("\n" + "="*60 + "\n")
    
    # Try GPU version if available
    if torch.cuda.is_available():
        print("Running GPU-accelerated version...")
        try:
            run_gpu_pipeline()
        except Exception as e:
            print(f"GPU processing failed: {e}")
            print("Falling back to CPU version")