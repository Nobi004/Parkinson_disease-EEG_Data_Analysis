import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings
warnings.filterwarnings('ignore')

class SimpleEEGProcessor:
    """Simple, stable EEG processor that just works"""
    
    def __init__(self, sampling_rate=512):
        self.fs = sampling_rate
        
        # Pre-compute filter
        nyquist = self.fs / 2
        self.sos_filter = butter(4, [0.5/nyquist, 50/nyquist], btype='band', output='sos')
    
    def process_subject(self, data):
        """Process single subject data"""
        try:
            # Clean data
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
            
            # Simple filtering
            filtered = sosfiltfilt(self.sos_filter, data, axis=1)
            
            # Extract basic features
            features = []
            
            # Mean and std for each channel
            features.extend(np.mean(filtered, axis=1))
            features.extend(np.std(filtered, axis=1))
            
            # Band powers (simple method)
            for ch in range(data.shape[0]):
                fft = np.abs(np.fft.rfft(filtered[ch]))
                features.extend([
                    np.sum(fft[1:4]),      # Delta
                    np.sum(fft[4:8]),      # Theta  
                    np.sum(fft[8:13]),     # Alpha
                    np.sum(fft[13:30]),    # Beta
                    np.sum(fft[30:50])     # Gamma
                ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error: {e}")
            return None

# Usage
processor = SimpleEEGProcessor()

# Process your data
n_subjects = 149
n_channels = 64
n_samples = 1024

all_features = []
for i in range(n_subjects):
    # Generate test data (replace with your real data)
    data = np.random.randn(n_channels, n_samples) * 50
    
    features = processor.process_subject(data)
    if features is not None:
        all_features.append(features)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{n_subjects} subjects")

features_matrix = np.vstack(all_features)
print(f"Final features shape: {features_matrix.shape}")