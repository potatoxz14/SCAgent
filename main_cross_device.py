import os
import glob
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    raise SystemExit("Please install TabPFN first:  pip install tabpfn")

try:
    from pca_utils import train_and_save_pca, load_and_transform_pca
    HAS_PCA = True
except ImportError:
    HAS_PCA = False
    print("Warning: pca_utils.py not found; using raw MiniRocket features instead of PCA.")

# ------------------------------- Configuration -------------------------------
RANDOM_STATE = 42
MINIROCKET_NUM_FEATURES = 10_000
PCA_TARGET_DIMENSION = 250
PCA_MODEL_PATH = './pca_cross_device.pkl'

TRAIN_DEVICES = ['13', '16pro']          # source devices used for training
EVAL_DEVICE = '14'                       # held-out evaluation device
TEST_DEVICE = '14pro'                    # held-out test device

TASKS = {
    'web': {
        'root': './web',
        'length': 640,
        'truncate': 'head',              # keep the first L samples
        'devices': {
            '13':    'iphone_13_web_with_diff',
            '14':    'iphone_14_web_with_diff',
            '14pro': 'iphone_14pro_web_with_diff',
            '16pro': 'iphone_16pro_web_with_diff',
        },
        'channels': ['fs_free_bytes', 'mem_wired', 'mem_zero_fill',
                     'net_en0_ibytes', 'net_en0_obytes', 'net_en0_opackets',
                     'latency_coreml', 'latency_metal', 'latency_filesystem', 'latency_cache'],
    },
    'app': {
        'root': './app',
        'length': 1000,
        'truncate': 'tail',              # keep the last L samples (launch signal is at the tail)
        'devices': {
            '13':    'os26_app_13_with_diff_v3_50',
            '14':    'os26_app_14_with_diff_v3_50',
            '14pro': 'os26_app_14pro_with_diff_v3_50',
            '16pro': 'os26_app_16pro_with_diff_v3_50',
        },
        'channels': ['en0_obytes', 'en0_ibytes', 'fs_delta_bytes', 'gpu_lat_ms',
                     'ane_lat_ms', 'mem_wire_cnt', 'jitter_us', 'mem_cow_faults'],
    },
}


# --------------------------------- Data loading ------------------------------
def load_trace(file_path, channels, length, truncate):
    """Load one CSV trace: keep the requested channels, per-channel z-score, and fix the length."""
    try:
        df = pd.read_csv(file_path, usecols=lambda c: c.strip() in channels)
        df.columns = df.columns.str.strip()
    except Exception:
        return None

    matrix = []
    for channel in channels:
        if channel not in df.columns:
            return None
        series = df[channel].values.astype(np.float32)
        if series.size == 0:
            return None
        mean, std = series.mean(), series.std()
        series = (series - mean) / (std + 1e-8)                    # per-channel z-score
        n = len(series)
        if n > length:
            series = series[:length] if truncate == 'head' else series[n - length:]
        elif n < length:
            pad = np.full(length - n, series[0], dtype=np.float32)
            series = np.concatenate([pad, series])
        matrix.append(series)
    return np.stack(matrix, axis=0)


def load_device(task, device_key):
    """Load every trace of one device for one task. Returns (X, y_str)."""
    cfg = TASKS[task]
    root = os.path.join(cfg['root'], cfg['devices'][device_key])
    X, y = [], []
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir) or class_name == '__warmup__':
            continue
        for f in sorted(glob.glob(os.path.join(class_dir, '*.csv'))):
            trace = load_trace(f, cfg['channels'], cfg['length'], cfg['truncate'])
            if trace is not None:
                X.append(trace)
                y.append(class_name)
    return np.stack(X).astype(np.float32), np.array(y)


# ----------------------------------- Pipeline --------------------------------
def reduce_dimension(features, fit):
    """PCA to PCA_TARGET_DIMENSION (fit + save on training, load + apply on evaluation)."""
    if not HAS_PCA:
        return features
    if fit:
        reduced, _ = train_and_save_pca(features, PCA_TARGET_DIMENSION, PCA_MODEL_PATH)
        return reduced
    return load_and_transform_pca(features, PCA_MODEL_PATH)


def evaluate(classifier, minirocket, encoder, X, y_str):
    """Accuracy on a held-out device (classes unseen during training are skipped)."""
    keep = np.isin(y_str, encoder.classes_)
    X, y_str = X[keep], y_str[keep]
    features = reduce_dimension(minirocket.transform(X), fit=False)
    return accuracy_score(encoder.transform(y_str), classifier.predict(features)), X.shape[0]


def run_task(task):
    cfg = TASKS[task]
    t0 = time.time()
    print(f"\n========== {task.upper()}  cross-device transfer "
          f"({len(cfg['channels'])} channels, L={cfg['length']}) ==========", flush=True)

    # 1) Load the source (training) devices and the two held-out devices.
    data = {d: load_device(task, d) for d in set(TRAIN_DEVICES + [EVAL_DEVICE, TEST_DEVICE])}
    X_train = np.concatenate([data[d][0] for d in TRAIN_DEVICES], axis=0)
    y_train_str = np.concatenate([data[d][1] for d in TRAIN_DEVICES], axis=0)

    encoder = LabelEncoder().fit(y_train_str)
    y_train = encoder.transform(y_train_str)
    print(f"  Train devices {TRAIN_DEVICES}: {X_train.shape[0]} traces, "
          f"{len(encoder.classes_)} classes", flush=True)

    # 2) MiniRocket -> PCA -> One-vs-Rest TabPFN, fitted on the source devices only.
    minirocket = MiniRocketMultivariate(num_kernels=MINIROCKET_NUM_FEATURES, random_state=RANDOM_STATE)
    F_train = reduce_dimension(minirocket.fit_transform(X_train), fit=True)

    device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    classifier = OneVsRestClassifier(TabPFNClassifier(device=device))
    classifier.fit(F_train, y_train)

    # 3) Evaluate on iPhone 14, then test on iPhone 14 Pro (purely inductive).
    eval_acc, n_eval = evaluate(classifier, minirocket, encoder, *data[EVAL_DEVICE])
    test_acc, n_test = evaluate(classifier, minirocket, encoder, *data[TEST_DEVICE])
    print(f"  EVAL  iPhone {EVAL_DEVICE}: {n_eval} traces | accuracy {eval_acc:.4f}", flush=True)
    print(f"  TEST  iPhone {TEST_DEVICE}: {n_test} traces | accuracy {test_acc:.4f}", flush=True)
    print(f"  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    for task_name in TASKS:
        run_task(task_name)
