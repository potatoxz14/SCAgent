import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("Error: tabpfn not found, please run pip install tabpfn")
    exit()

try:
    from pca_utils import train_and_save_pca, load_and_transform_pca
except ImportError:
    print("Warning: pca_utils.py not found")
    pass


CSV_ROOT_FOLDER = '' 
ORIGINAL_TARGET_LENGTH = 640
ORIGINAL_FREQ = 100

TARGET_COLUMN_INDICES = [2, 6, 12, 13, 15, 24, 25, 26, 27]
MINIROCKET_NUM_FEATURES = 10_000

TEST_NOISE_RATIOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

def load_and_standardize_ts_base(file_path, target_length, column_indices):
    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    standardized_channels = []
    
    for index in column_indices:
        if index >= df.shape[1]:
            continue

        ts = df.iloc[:, index].values.astype(np.float32)
        current_length = len(ts)

        if current_length > target_length:
            start_index = current_length - target_length
            ts_final = ts[start_index:]
        elif current_length < target_length:
            padding_needed = target_length - current_length
            if index < 24:
                fill_value = 0.0
            else:
                fill_value = ts[0] if current_length > 0 else 0.0
            padding_array = np.full(padding_needed, fill_value, dtype=np.float32)
            ts_final = np.concatenate((padding_array, ts))
        else:
            ts_final = ts
            
        standardized_channels.append(ts_final)

    if not standardized_channels:
        return None

    return np.stack(standardized_channels, axis=0)

def add_gaussian_noise(X, noise_ratio):
    if noise_ratio <= 0:
        return X
    
    channel_stds = np.std(X, axis=(0, 2), keepdims=True)
    
    channel_stds[channel_stds == 0] = 1.0 
    
    noise = np.random.normal(loc=0.0, scale=channel_stds * noise_ratio, size=X.shape)
    
    return X + noise

if __name__ == '__main__':
    
    results = {}

    print(f"Noise Ratios: {TEST_NOISE_RATIOS}")
    print("==================================================")

    all_series = []
    all_labels = []
    
    if not os.path.exists(CSV_ROOT_FOLDER):
        print(f"Error:  {CSV_ROOT_FOLDER} does not exist")
        exit()

    for label_name in os.listdir(CSV_ROOT_FOLDER):
        label_dir = os.path.join(CSV_ROOT_FOLDER, label_name)
        if not os.path.isdir(label_dir):
            continue
        
        for csv_file in os.listdir(label_dir):
            if not csv_file.endswith('.csv'):
                continue
            
            file_path = os.path.join(label_dir, csv_file)
            
            ts_data = load_and_standardize_ts_base(
                file_path, 
                target_length=ORIGINAL_TARGET_LENGTH, 
                column_indices=TARGET_COLUMN_INDICES
            )
            
            if ts_data is not None:
                all_series.append(ts_data)
                all_labels.append(label_name)

    if len(all_series) == 0:
        print("Error: No data loaded!")
        exit()

    X_original = np.stack(all_series, axis=0) # (N, Channels, Length)
    y = np.array(all_labels)
    
    print(f"shape: {X_original.shape}")
    print("==================================================")


    for ratio in TEST_NOISE_RATIOS:
        print(f"\n[Testing Noise Ratio: {ratio} (Signal * {ratio})]")
        

        if ratio > 0:
            print("  -> Injecting Gaussian noise...")
            X_noisy = add_gaussian_noise(X_original, ratio)
        else:
            print("  -> Using clean data.")
            X_noisy = X_original

        X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
        
        minirocket = MiniRocketMultivariate(num_kernels=MINIROCKET_NUM_FEATURES, random_state=42)
        try:
            X_train_features = minirocket.fit_transform(X_train)
            X_test_features = minirocket.transform(X_test)
        except Exception as e:
            print(f"  -> MiniRocket Error: {e}")
            continue

        # 2.4 PCA
        from sklearn.decomposition import PCA
        n_components = min(250, X_train_features.shape[0], X_train_features.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_reduced = pca.fit_transform(X_train_features)
        X_test_reduced = pca.transform(X_test_features)
        
        # 2.5 TabPFN 
        try:
            classifier = OneVsRestClassifier(TabPFNClassifier(device='cuda'))
        except:
            print("  -> CUDA not available, using CPU.")
            classifier = OneVsRestClassifier(TabPFNClassifier(device='cpu'))
            
        classifier.fit(X_train_reduced, y_train)
        y_pred = classifier.predict(X_test_reduced)
        
        acc = accuracy_score(y_test, y_pred)
        results[ratio] = acc
        print(f"  -> ✅ Accuracy with Noise {ratio}: {acc:.4f}")

    print("\n==============================================")
    print("Final Noise Robustness Report:")
    print("Ratio 0.1 means noise_std = 0.1 * feature_std")
    print("----------------------------------------------")
    for ratio in sorted(results.keys()):
        print(f"Noise Ratio: {ratio:<5} | Accuracy: {results[ratio]:.4f}")
    print("==============================================")