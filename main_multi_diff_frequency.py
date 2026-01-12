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
    pass

CSV_ROOT_FOLDER = '' 
ORIGINAL_TARGET_LENGTH = 1000
ORIGINAL_FREQ = 100

TARGET_COLUMN_INDICES = [11, 14, 15, 16, 3, 4, 2, 8, 21]
MINIROCKET_NUM_FEATURES = 10_000

TEST_FREQUENCIES = [5, 10, 20, 50, 100]

def load_and_standardize_ts_mixed_sampling(file_path, target_length, column_indices, step):
    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    standardized_channels = []
    
    raw_len = len(df)
    n_samples = raw_len // step
    if n_samples == 0:
        return None 
        
    limit_idx = n_samples * step

    for index in column_indices:
        if index >= df.shape[1]:
            continue
        
        ts_raw = df.iloc[:limit_idx, index].values.astype(np.float32)
        
        if step > 1:
            ts_reshaped = ts_raw.reshape(n_samples, step)
            
            if index < 24:
                ts_downsampled = ts_reshaped.sum(axis=1)
            else:
                ts_downsampled = ts_reshaped[:, 0]
        else:
            ts_downsampled = ts_raw

        current_length = len(ts_downsampled)
        
        if current_length > target_length:
            start_index = current_length - target_length
            ts_final = ts_downsampled[start_index:]
        elif current_length < target_length:
            padding_needed = target_length - current_length
            
            if index < 24:
                fill_value = 0.0
            else:
                fill_value = ts_downsampled[0] if current_length > 0 else 0.0
                
            padding_array = np.full(padding_needed, fill_value, dtype=np.float32)
            ts_final = np.concatenate((padding_array, ts_downsampled))
        else:
            ts_final = ts_downsampled
            
        standardized_channels.append(ts_final)

    if not standardized_channels:
        return None

    return np.stack(standardized_channels, axis=0)

if __name__ == '__main__':
    
    results = {}

    print(f": {TEST_FREQUENCIES} Hz")
    print(f"Target Columns: {TARGET_COLUMN_INDICES}")
    print("==================================================")

    for freq in TEST_FREQUENCIES:
        step = int(ORIGINAL_FREQ / freq)
        current_target_length = int(ORIGINAL_TARGET_LENGTH / step)
        
        if current_target_length < 16: 
            print(f"Skipping {freq}Hz: Target length {current_target_length} too short.")
            continue

        print(f"\n[Testing Frequency: {freq} Hz]")
        print(f"  -> Downsample Step: {step}")
        print(f"  -> Mode: Sum(Diffs) + DirectSample(States)")
        print(f"  -> New Target Length: {current_target_length}")
        
        all_series = []
        all_labels = []
        
        if not os.path.exists(CSV_ROOT_FOLDER):
            print(f"Error: {CSV_ROOT_FOLDER} does not exist")
            continue

        for label_name in os.listdir(CSV_ROOT_FOLDER):
            label_dir = os.path.join(CSV_ROOT_FOLDER, label_name)
            if not os.path.isdir(label_dir):
                continue
            
            for csv_file in os.listdir(label_dir):
                if not csv_file.endswith('.csv'):
                    continue
                
                file_path = os.path.join(label_dir, csv_file)
                
                ts_data = load_and_standardize_ts_mixed_sampling(
                    file_path, 
                    target_length=current_target_length, 
                    column_indices=TARGET_COLUMN_INDICES,
                    step=step
                )
                
                if ts_data is not None:
                    all_series.append(ts_data)
                    all_labels.append(label_name)

        if len(all_series) == 0:
            print("  -> No data loaded!")
            continue

        X = np.stack(all_series, axis=0)
        y = np.array(all_labels)
        
        print(f"  -> Data Shape: {X.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        minirocket = MiniRocketMultivariate(num_kernels=MINIROCKET_NUM_FEATURES)
        try:
            X_train_features = minirocket.fit_transform(X_train)
            X_test_features = minirocket.transform(X_test)
        except Exception as e:
            print(f"  -> MiniRocket Error: {e}")
            continue

        from sklearn.decomposition import PCA
        n_components = min(250, X_train_features.shape[0], X_train_features.shape[1])
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train_features)
        X_test_reduced = pca.transform(X_test_features)
        
        try:
            classifier = OneVsRestClassifier(TabPFNClassifier(device='cuda'))
        except:
            print("  -> CUDA not available, using CPU.")
            classifier = OneVsRestClassifier(TabPFNClassifier(device='cpu'))
            
        classifier.fit(X_train_reduced, y_train)
        y_pred = classifier.predict(X_test_reduced)
        
        acc = accuracy_score(y_test, y_pred)
        results[freq] = acc
        print(f"  -> ✅ Accuracy at {freq}Hz: {acc:.4f}")

    print("\n==============================================")
    print("Final Frequency Evaluation Report:")
    for freq in sorted(results.keys(), reverse=True):
        print(f"Freq: {freq} Hz | Accuracy: {results[freq]:.4f}")
    print("==============================================")

