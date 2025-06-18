# Step-by-Step Guide to Improve EEG Parkinson's Detection Accuracy

## Current Baseline Assessment

First, let's establish your current performance metrics:

```python
# Run this to get baseline metrics
def evaluate_current_model(model, test_loader):
    """Get comprehensive baseline metrics"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Control', 'Parkinson']))
    
    return all_preds, all_labels, all_probs
```

## Step 1: Data Quality Improvements (Expected: +5-10% accuracy)

### 1.1 Advanced Artifact Removal

```python
def improve_preprocessing(preprocessor):
    """Enhanced preprocessing with better artifact removal"""
    
    # Add adaptive filtering
    preprocessor.use_adaptive_filter = True
    
    # Add eye blink removal
    preprocessor.remove_eye_blinks = True
    
    # Add muscle artifact removal
    preprocessor.remove_muscle_artifacts = True
    
    # Use robust statistics
    preprocessor.use_robust_stats = True
    
    return preprocessor
```

### 1.2 Data Augmentation (Critical for Small Datasets)

```python
class EEGAugmentation:
    """Data augmentation specifically for EEG"""
    
    def __init__(self):
        self.augmentations = {
            'noise': self.add_gaussian_noise,
            'scale': self.random_scaling,
            'shift': self.time_shift,
            'flip': self.channel_flip,
            'mixup': self.mixup
        }
    
    def add_gaussian_noise(self, data, noise_level=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        return data + noise
    
    def random_scaling(self, data, scale_range=(0.9, 1.1)):
        """Random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return data * scale
    
    def time_shift(self, data, max_shift=50):
        """Random time shifting"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(data, shift, axis=-1)
    
    def channel_flip(self, data, prob=0.5):
        """Randomly flip polarity of channels"""
        if np.random.random() < prob:
            flip_mask = np.random.choice([-1, 1], size=(data.shape[0], 1))
            return data * flip_mask
        return data
    
    def mixup(self, data1, data2, alpha=0.2):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        return lam * data1 + (1 - lam) * data2
    
    def augment_batch(self, batch, labels, n_augment=2):
        """Augment a batch of data"""
        augmented_data = []
        augmented_labels = []
        
        for data, label in zip(batch, labels):
            # Original data
            augmented_data.append(data)
            augmented_labels.append(label)
            
            # Augmented versions
            for _ in range(n_augment):
                aug_data = data.copy()
                
                # Apply random augmentations
                for aug_name, aug_func in self.augmentations.items():
                    if np.random.random() < 0.5:  # 50% chance
                        if aug_name != 'mixup':
                            aug_data = aug_func(aug_data)
                
                augmented_data.append(aug_data)
                augmented_labels.append(label)
        
        return np.array(augmented_data), np.array(augmented_labels)
```

## Step 2: Feature Engineering Improvements (Expected: +5-15% accuracy)

### 2.1 Parkinson's-Specific Features

```python
def extract_parkinsons_features(epochs, fs=512):
    """Extract features specifically relevant to Parkinson's"""
    features = []
    
    for epoch in epochs:
        epoch_features = []
        
        # 1. Beta band (13-30 Hz) features - key for PD
        for ch in range(epoch.shape[0]):
            freqs, psd = welch(epoch[ch], fs)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            # Beta power
            beta_power = np.sum(psd[beta_mask])
            epoch_features.append(beta_power)
            
            # Beta peak frequency
            beta_peak = freqs[beta_mask][np.argmax(psd[beta_mask])]
            epoch_features.append(beta_peak)
            
            # Beta bandwidth
            beta_bw = np.sum(psd[beta_mask] > 0.5 * np.max(psd[beta_mask]))
            epoch_features.append(beta_bw)
        
        # 2. Tremor band (4-6 Hz) features
        tremor_powers = []
        for ch in range(epoch.shape[0]):
            tremor_mask = (freqs >= 4) & (freqs <= 6)
            tremor_power = np.sum(psd[tremor_mask])
            tremor_powers.append(tremor_power)
        
        epoch_features.extend(tremor_powers)
        epoch_features.append(np.mean(tremor_powers))
        epoch_features.append(np.std(tremor_powers))
        
        # 3. Cortical slowing (theta/alpha ratio)
        for ch in range(epoch.shape[0]):
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            
            theta_power = np.sum(psd[theta_mask])
            alpha_power = np.sum(psd[alpha_mask])
            
            ratio = theta_power / (alpha_power + 1e-10)
            epoch_features.append(ratio)
        
        # 4. Inter-hemispheric asymmetry (important for PD)
        # Compare left vs right hemisphere channels
        left_channels = [0, 2, 4, 6]  # Example indices
        right_channels = [1, 3, 5, 7]
        
        for band_low, band_high in [(13, 30), (4, 8), (8, 13)]:
            left_power = 0
            right_power = 0
            
            for l_ch, r_ch in zip(left_channels, right_channels):
                if l_ch < epoch.shape[0] and r_ch < epoch.shape[0]:
                    # Left hemisphere
                    freqs_l, psd_l = welch(epoch[l_ch], fs)
                    band_mask = (freqs_l >= band_low) & (freqs_l <= band_high)
                    left_power += np.sum(psd_l[band_mask])
                    
                    # Right hemisphere
                    freqs_r, psd_r = welch(epoch[r_ch], fs)
                    band_mask = (freqs_r >= band_low) & (freqs_r <= band_high)
                    right_power += np.sum(psd_r[band_mask])
            
            asymmetry = (left_power - right_power) / (left_power + right_power + 1e-10)
            epoch_features.append(asymmetry)
        
        # 5. Connectivity features in motor cortex
        motor_channels = [10, 11, 12, 13]  # C3, C4, etc.
        coherences = []
        
        for i in range(len(motor_channels)-1):
            for j in range(i+1, len(motor_channels)):
                if motor_channels[i] < epoch.shape[0] and motor_channels[j] < epoch.shape[0]:
                    f, Cxy = coherence(epoch[motor_channels[i]], 
                                     epoch[motor_channels[j]], fs)
                    # Beta band coherence
                    beta_mask = (f >= 13) & (f <= 30)
                    coherences.append(np.mean(Cxy[beta_mask]))
        
        epoch_features.extend(coherences)
        features.append(epoch_features)
    
    return np.array(features)
```

### 2.2 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def select_best_features(X_train, y_train, X_test, n_features=100):
    """Select most informative features"""
    
    # Method 1: Univariate selection
    selector1 = SelectKBest(f_classif, k=n_features)
    selector1.fit(X_train, y_train)
    scores1 = selector1.scores_
    
    # Method 2: Mutual information
    selector2 = SelectKBest(mutual_info_classif, k=n_features)
    selector2.fit(X_train, y_train)
    scores2 = selector2.scores_
    
    # Method 3: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    scores3 = rf.feature_importances_
    
    # Combine scores (rank-based fusion)
    combined_ranks = np.argsort(scores1) + np.argsort(scores2) + np.argsort(scores3)
    best_features = np.argsort(combined_ranks)[-n_features:]
    
    return X_train[:, best_features], X_test[:, best_features], best_features
```

## Step 3: Model Architecture Improvements (Expected: +5-10% accuracy)

### 3.1 Ensemble Learning

```python
class EnsembleModel(nn.Module):
    """Ensemble of different architectures"""
    
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(F.softmax(model(x), dim=1))
        
        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        output = sum(w * out for w, out in zip(weights, outputs))
        
        return output

# Create ensemble
ensemble = EnsembleModel([
    create_model('eegnet', config),
    create_model('transformer', config),
    create_model('hybrid', config)
])
```

### 3.2 Attention Mechanisms

```python
class AttentionEEGNet(nn.Module):
    """EEGNet with channel and temporal attention"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply channel attention
        ch_att = self.channel_attention(x.mean(dim=2))
        x = x * ch_att.unsqueeze(2)
        
        # Apply temporal attention
        temp_att = self.temporal_attention(x.mean(dim=1, keepdim=True))
        x = x * temp_att
        
        return self.base_model(x)
```

## Step 4: Training Strategy Improvements (Expected: +5-10% accuracy)

### 4.1 Advanced Training Techniques

```python
def train_with_advanced_techniques(model, train_loader, val_loader, config):
    """Training with modern techniques"""
    
    # 1. Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01
    )
    
    # 2. Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # 3. Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 4. Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. Gradient accumulation
    accumulation_steps = 4
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            
            # Gradient accumulation
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            scheduler.step()
        
        # Validation with early stopping
        val_acc = validate(model, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_val_acc
```

### 4.2 Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(data, labels, config, n_folds=5):
    """Stratified k-fold cross-validation"""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Data augmentation on training set only
        augmenter = EEGAugmentation()
        X_train_aug, y_train_aug = augmenter.augment_batch(X_train, y_train)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_aug), 
            torch.LongTensor(y_train_aug)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
        
        # Train model
        model = create_model('eegnet', config).to(config.DEVICE)
        fold_score = train_with_advanced_techniques(model, train_loader, 
                                                   val_loader, config)
        fold_scores.append(fold_score)
    
    print(f"\nCross-validation scores: {fold_scores}")
    print(f"Mean CV score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    return fold_scores
```

## Step 5: Hyperparameter Optimization (Expected: +5-10% accuracy)

### 5.1 Automated Hyperparameter Search

```python
import optuna

def objective(trial):
    """Optuna objective function"""
    
    # Hyperparameters to optimize
    params = {
        'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
        'n_filters': trial.suggest_int('n_filters', 16, 64, step=16),
    }
    
    # Train model with these parameters
    model = create_model_with_params(params)
    accuracy = train_and_evaluate(model, params)
    
    return accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

## Step 6: Post-Processing and Calibration (Expected: +2-5% accuracy)

### 6.1 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_predictions(model, X_val, y_val):
    """Calibrate model predictions"""
    
    # Get raw predictions
    model.eval()
    with torch.no_grad():
        raw_probs = torch.softmax(model(torch.FloatTensor(X_val)), dim=1).numpy()
    
    # Calibrate
    calibrator = CalibratedClassifierCV(cv=3, method='isotonic')
    calibrator.fit(raw_probs[:, 1].reshape(-1, 1), y_val)
    
    return calibrator
```

### 6.2 Test-Time Augmentation

```python
def predict_with_tta(model, data, n_augmentations=10):
    """Test-time augmentation for better predictions"""
    
    augmenter = EEGAugmentation()
    predictions = []
    
    for _ in range(n_augmentations):
        # Light augmentation
        aug_data = augmenter.add_gaussian_noise(data, noise_level=0.02)
        aug_data = augmenter.random_scaling(aug_data, scale_range=(0.95, 1.05))
        
        with torch.no_grad():
            pred = torch.softmax(model(torch.FloatTensor(aug_data)), dim=1)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

## Implementation Checklist

1. **Week 1: Data Quality**
   - [ ] Implement advanced artifact removal
   - [ ] Add data augmentation
   - [ ] Verify data quality improvements

2. **Week 2: Feature Engineering**
   - [ ] Extract Parkinson's-specific features
   - [ ] Implement feature selection
   - [ ] Combine with existing features

3. **Week 3: Model Architecture**
   - [ ] Create ensemble model
   - [ ] Add attention mechanisms
   - [ ] Test different architectures

4. **Week 4: Training Optimization**
   - [ ] Implement advanced training techniques
   - [ ] Run cross-validation
   - [ ] Optimize hyperparameters

5. **Week 5: Final Improvements**
   - [ ] Calibrate predictions
   - [ ] Add test-time augmentation
   - [ ] Final evaluation

## Expected Results

Starting from a baseline of ~85% accuracy, implementing all these improvements should get you to:

- Data quality improvements: +5-10% → 90-95%
- Feature engineering: +5-15% → 95-99%
- Model improvements: +5-10% → 97-99%
- Training optimization: +5-10% → 98-99%
- Post-processing: +2-5% → 99%+

**Final expected accuracy: 95-99%**

## Debugging Low Accuracy

If accuracy is still low, check:

1. **Data Issues**
   ```python
   # Check class balance
   print(f"Class distribution: {np.bincount(labels)}")
   
   # Check data quality
   print(f"NaN values: {np.isnan(data).sum()}")
   print(f"Inf values: {np.isinf(data).sum()}")
   ```

2. **Feature Quality**
   ```python
   # Visualize features
   import seaborn as sns
   sns.heatmap(np.corrcoef(features.T), cmap='coolwarm')
   ```

3. **Model Training**
   ```python
   # Plot training curves
   plt.plot(train_losses, label='Train')
   plt.plot(val_losses, label='Val')
   plt.legend()
   ```

Remember: Start with one improvement at a time and measure the impact before moving to the next!