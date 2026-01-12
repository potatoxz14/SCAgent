import pandas as pd
import numpy as np
import os
import joblib
import time
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
    print("Error: pca_utils.py not found.")
    exit()



CSV_ROOT_FOLDER = ''
TARGET_LENGTH = 640


TARGET_COLUMN_INDICES = [2, 6, 12, 13, 15, 24, 25, 26, 27]  #ours optimal web
# TARGET_COLUMN_INDICES = [2, 24, 25, 26, 27] # pure new web 
# TARGET_COLUMN_INDICES = [3, 5, 6, 9, 12, 13, 14, 15, 20]  #2023 web
# TARGET_COLUMN_INDICES = [3, 4, 7, 8, 12, 13]  #2018 web
# --- MINIROCKET Configuration ---
# TARGET_COLUMN_INDICES = [2, ]  #2018 web
MINIROCKET_NUM_FEATURES = 10_000
MINIROCKET_MAX_DILATIONS = 32
MINIROCKET_PARAMS_PATH = './minirocket_params.joblib'

# --- PCA Configuration ---
PCA_TARGET_DIMENSION = 250
PCA_MODEL_PATH = './pca_minirocket_to_100_model.pkl'

# --- Evaluation Configuration ---
TEST_SIZE = 0.2  
RANDOM_STATE = 42
# ---------------------




def load_and_standardize_ts(file_path, target_length, column_indices):
    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    standardized_channels = []

    for index in column_indices:
        if index >= df.shape[1]:
            print(f"Warning: Column index {index} out of bounds for {file_path}. Skipping.")
            continue
            
        ts = df.iloc[:, index].values.astype(np.float32)
        current_length = len(ts)

        if current_length > target_length:
            start_index = current_length - target_length
            ts_standardized = ts[start_index:]
        elif current_length < target_length:
            padding_needed = target_length - current_length
            if current_length > 0:
                fill_value = ts[0]
            else:
                fill_value = 0.0
            padding_array = np.full(padding_needed, fill_value, dtype=np.float32)
            ts_standardized = np.concatenate((padding_array, ts))
        else:
            ts_standardized = ts
        
        standardized_channels.append(ts_standardized)

    if not standardized_channels:
        return None
    return np.stack(standardized_channels, axis=0) 

# -------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------

if __name__ == '__main__':
    
    # --- 1. Load and Standardize Data ---
    print("--- 1. Data Loading and Preprocessing (Multivariate) ---")
    all_series = []
    all_labels = [] 
    
    all_subdirs = [d for d in os.listdir(CSV_ROOT_FOLDER) if os.path.isdir(os.path.join(CSV_ROOT_FOLDER, d))]
    all_subdirs.sort()

    for label_index, subdir_name in enumerate(all_subdirs):
        current_label = label_index + 1 
        subdir_path = os.path.join(CSV_ROOT_FOLDER, subdir_name)
        
        if os.path.isdir(subdir_path): 
            for filename in os.listdir(subdir_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(subdir_path, filename)
                    
                    standard_ts = load_and_standardize_ts(file_path, TARGET_LENGTH, TARGET_COLUMN_INDICES)
                    
                    if standard_ts is not None:
                        all_series.append(standard_ts)
                        all_labels.append(current_label) 

    X_raw = np.stack(all_series, axis=0)
    y_raw = np.array(all_labels)

    print(f"Loaded raw data shape (X_raw): {X_raw.shape}") # Expecting (N, C, L)
    print(f"Loaded raw labels shape (y_raw): {y_raw.shape}")

    X_raw = X_raw.astype(np.float32)

    print("\n--- 1.5 Splitting Data into Training and Testing Sets ---")
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, 
        y_raw, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_raw 
    )

    print(f"Training data shape: {X_train_raw.shape}")
    print(f"Testing data shape: {X_test_raw.shape}")

    print("\n⏱️  Starting Training Timer...")
    training_start_time = time.time()
    # --- 2. MINIROCKET Feature Generation ---
    print("\n--- 2. MINIROCKET Feature Generation ---")


    print(f"Fitting MiniRocket on data with {X_train_raw.shape[1]} channels...")
    
    minirocket = MiniRocketMultivariate(num_kernels=10000)


    X_train_features = minirocket.fit_transform(X_train_raw)
    X_test_features = minirocket.transform(X_test_raw)

    print("\n--- 3. ROCKET  ---")
    print(f"ROCKET Train Feature shape: {X_train_features.shape}")
    print(f"ROCKET Test Feature shape: {X_test_features.shape}")

    print("\n--- 4. PCA Feature Reduction for TabPFN (Using cuML GPU) ---")

    try:
        X_train_reduced, _ = train_and_save_pca(
            X=X_train_features, 
            n_components=PCA_TARGET_DIMENSION, 
            model_path=PCA_MODEL_PATH
        )
    except Exception as e:
        X_train_reduced = X_train_features
        

    X_test_reduced = load_and_transform_pca(
        X_new=X_test_features, 
        model_path=PCA_MODEL_PATH
    )

    print(f"Final Train shape for TabPFN: {X_train_reduced.shape}")
    print(f"Final Test shape for TabPFN: {X_test_reduced.shape}")


    print("\n--- 6. TabPFN Training and Evaluation ---")
    
    try:
        base_classifier = TabPFNClassifier(device='cuda') 
        print("TabPFN Base Classifier initialized (Device: cuda/GPU).")
    except RuntimeError:
        base_classifier = TabPFNClassifier(device='cpu')
        print("TabPFN Base Classifier initialized (Device: cpu).")

    classifier = OneVsRestClassifier(base_classifier)
    print(f"Wrapped TabPFN with OneVsRestClassifier for {np.unique(y_train).size} classes.")

    classifier.fit(X_train_reduced, y_train)
    print("TabPFN (OvR) training complete.")
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print("\n" + "="*40)
    print(f"✅ Training Complete!")
    print(f"⏱️  Total Training Time: {total_training_time:.4f} seconds")
    print("="*40)

    y_pred = classifier.predict(X_test_reduced)
    print("TabPFN (OvR) prediction complete.")
    predicting_end_time = time.time()

    total_time = predicting_end_time - training_start_time
    print("\n" + "="*40)
    print(f"✅ predicting Complete!")
    print(f"⏱️  Total Training Time: {total_time:.4f} seconds")
    print("="*40)
    accuracy = accuracy_score(y_test, y_pred)        
    
    print("\n==============================================")
    print(f"✅ Final Classification Accuracy on Test Set: {accuracy:.4f}")
    print("==============================================")
