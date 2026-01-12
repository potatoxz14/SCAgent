import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate


try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("Error: pip install tabpfn")
    exit()

try:
    from pca_utils import train_and_save_pca, load_and_transform_pca
except ImportError:
    print("Warning: pca_utils.py missing.")
    pass


FOLDER_IPHONE_13 = '/data_web_iphone_13_v2_with_diff'
FOLDER_IPHONE_14 = '/data_web_iphone_14_v2_with_diff'


TRAIN_SLICE = slice(0, 40)      


TEST_SLICE = slice(40, None)    


TARGET_LENGTH = 640

TARGET_COLUMN_INDICES = [2, 24, 25, 26, 27]

MINIROCKET_NUM_FEATURES = 10_000
PCA_TARGET_DIMENSION = 250
PCA_MODEL_PATH = './pca_dual_test_model.pkl'
RANDOM_STATE = 42
# ==========================================

def load_and_standardize_ts(file_path, target_length, column_indices):
    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    standardized_channels = []
    for index in column_indices:
        if index >= df.shape[1]: continue
        ts = df.iloc[:, index].values.astype(np.float32)
        current_length = len(ts)
        if current_length > target_length:
            ts_standardized = ts[current_length - target_length:]
        elif current_length < target_length:
            padding = np.full(target_length - current_length, ts[0] if current_length > 0 else 0.0, dtype=np.float32)
            ts_standardized = np.concatenate((padding, ts))
        else:
            ts_standardized = ts
        standardized_channels.append(ts_standardized)

    if not standardized_channels: return None
    return np.stack(standardized_channels, axis=0)

def load_dataset_with_slice(root_folder, target_length, col_indices, sample_slice, dataset_name="Dataset"):
    print(f"\n--- Loading {dataset_name} from: {os.path.basename(root_folder)} ---")
    print(f"    Slice Range: {sample_slice}")
    
    if not os.path.exists(root_folder):
        print(f"Error: Path not found {root_folder}")
        return None, None

    all_series = []
    all_labels_str = []
    subdirs = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
    
    if not subdirs: return None, None

    total_files = 0
    for subdir_name in subdirs:
        subdir_path = os.path.join(root_folder, subdir_name)
        files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.csv')])
        
        files_to_load = files[sample_slice]
        
        for filename in files_to_load:
            ts = load_and_standardize_ts(os.path.join(subdir_path, filename), target_length, col_indices)
            if ts is not None:
                all_series.append(ts)
                all_labels_str.append(subdir_name)
                total_files += 1

    if total_files == 0:
        print(f"Warning: No files loaded for {dataset_name}.")
        return None, None

    print(f"Loaded {total_files} samples.")
    return np.stack(all_series, axis=0).astype(np.float32), np.array(all_labels_str)

# -------------------------------------------------------------
if __name__ == '__main__':
    
    X_train_13, y_train_13 = load_dataset_with_slice(
        FOLDER_IPHONE_13, TARGET_LENGTH, TARGET_COLUMN_INDICES, 
        sample_slice=TRAIN_SLICE, dataset_name="Train Part 1 (iPhone 13)"
    )
    
    X_train_14, y_train_14 = load_dataset_with_slice(
        FOLDER_IPHONE_14, TARGET_LENGTH, TARGET_COLUMN_INDICES, 
        sample_slice=TRAIN_SLICE, dataset_name="Train Part 2 (iPhone 14)"
    )

    X_test_13, y_test_str_13 = load_dataset_with_slice(
        FOLDER_IPHONE_13, TARGET_LENGTH, TARGET_COLUMN_INDICES, 
        sample_slice=TEST_SLICE, dataset_name="TEST SET 1 (iPhone 13 Last 10)"
    )

    X_test_14, y_test_str_14 = load_dataset_with_slice(
        FOLDER_IPHONE_14, TARGET_LENGTH, TARGET_COLUMN_INDICES, 
        sample_slice=TEST_SLICE, dataset_name="TEST SET 2 (iPhone 14 Last 10)"
    )

    if any(x is None for x in [X_train_13, X_train_14, X_test_13, X_test_14]):
        print("Error: One of the datasets failed to load.")
        exit()

    print("\n--- Merging Training Sets ---")
    X_train = np.concatenate([X_train_13, X_train_14], axis=0)
    y_train_str = np.concatenate([y_train_13, y_train_14], axis=0)
    
    print(f"Final Train Shape: {X_train.shape}")
    print(f"Test 13 Shape:     {X_test_13.shape}")
    print(f"Test 14 Shape:     {X_test_14.shape}")

    print("\n--- Label Encoding ---")
    le = LabelEncoder()
    le.fit(y_train_str) # Fit on all training data
    

    y_train = le.transform(y_train_str)
    
    def safe_transform(y_str, name):
        mask = np.isin(y_str, le.classes_)
        if not np.all(mask):
            print(f"Warning: Dropping unknown classes from {name}")
            return y_str[mask], mask
        return y_str, mask

    y_test_str_13, mask_13 = safe_transform(y_test_str_13, "Test 13")
    X_test_13 = X_test_13[mask_13]
    y_test_13 = le.transform(y_test_str_13)

    y_test_str_14, mask_14 = safe_transform(y_test_str_14, "Test 14")
    X_test_14 = X_test_14[mask_14]
    y_test_14 = le.transform(y_test_str_14)

    print("Labels encoded successfully.")

    print("\n--- MiniRocket Feature Extraction ---")
    minirocket = MiniRocketMultivariate(num_kernels=MINIROCKET_NUM_FEATURES, random_state=RANDOM_STATE)
    

    X_train_feat = minirocket.fit_transform(X_train)
    

    X_test_feat_13 = minirocket.transform(X_test_13)
    X_test_feat_14 = minirocket.transform(X_test_14)

    print("\n--- PCA Reduction ---")
    try:
        X_train_pca, _ = train_and_save_pca(X_train_feat, PCA_TARGET_DIMENSION, PCA_MODEL_PATH)
        X_test_pca_13 = load_and_transform_pca(X_test_feat_13, PCA_MODEL_PATH)
        X_test_pca_14 = load_and_transform_pca(X_test_feat_14, PCA_MODEL_PATH)
    except:
        print("PCA skipped.")
        X_train_pca = X_train_feat
        X_test_pca_13 = X_test_feat_13
        X_test_pca_14 = X_test_feat_14

    print("\n--- Training TabPFN ---")
    clf = OneVsRestClassifier(TabPFNClassifier(device='cuda' if os.system("nvidia-smi") == 0 else 'cpu'))
    
    clf.fit(X_train_pca, y_train)
    print("\n" + "="*40)
    print(">>> EVALUATION 1: iPhone 13 (Held-out set)")
    print("="*40)
    y_pred_13 = clf.predict(X_test_pca_13)
    acc_13 = accuracy_score(y_test_13, y_pred_13)
    print(f"🚀 Accuracy on iPhone 13: {acc_13:.4f}")
    # print(classification_report(y_test_13, y_pred_13, target_names=[str(c) for c in le.classes_]))

    print("\n" + "="*40)
    print(">>> EVALUATION 2: iPhone 14 (Held-out set)")
    print("="*40)
    y_pred_14 = clf.predict(X_test_pca_14)
    acc_14 = accuracy_score(y_test_14, y_pred_14)
    print(f"🚀 Accuracy on iPhone 14: {acc_14:.4f}")

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print(f"Train Size: {len(X_train)} samples (Mixed 13 & 14)")
    print(f"Test 13 Accuracy: {acc_13:.4f}")
    print(f"Test 14 Accuracy: {acc_14:.4f}")
    print("="*40)