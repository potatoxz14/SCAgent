import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "./data_app_all_v2_with_diff"
SEQUENCE_LENGTH = 1000

PAA_WINDOW = 5       # p
ALPHABET_SIZE = 5    # a
WORD_LENGTH = 5     # w 


COLUMN_CONFIG = {
    'en0_obytes': False,       
    'en0_ibytes': False,       
    'mem_wire_cnt': False,     
    'mem_cow_faults': False,     
    'fs_delta_bytes': False,   
    'ane_lat_ms': False,       
    'gpu_lat_ms': False,       
    'jitter_us': False,        
}

class SaxBopExtractor:
    def __init__(self, alphabet_size, word_length, paa_window):
        self.a = alphabet_size
        self.w = word_length
        self.p = paa_window
        self.breakpoints = stats.norm.ppf(np.linspace(1./self.a, 1-1./self.a, self.a-1))
        chars = [chr(97 + i) for i in range(self.a)]
        self.vocab = [''.join(p) for p in itertools.product(chars, repeat=self.w)]
        self.vocab_map = {w: i for i, w in enumerate(self.vocab)}

    def transform_channel(self, time_series):
        # 1. Z-Normalize
        std = np.std(time_series)
        if std == 0: return np.zeros(len(self.vocab))
        ts_norm = (time_series - np.mean(time_series)) / std
        
        # 2. PAA
        n_seg = len(ts_norm) // self.p
        if n_seg == 0: return np.zeros(len(self.vocab))
        ts_paa = ts_norm[:n_seg*self.p].reshape(n_seg, self.p).mean(axis=1)
        
        # 3. SAX Discretization
        indices = np.digitize(ts_paa, self.breakpoints)
        sax_str = "".join([chr(97 + i) for i in indices])
        
        # 4. BOP (Bag of Patterns)
        counts = np.zeros(len(self.vocab))
        words = [sax_str[i:i+self.w] for i in range(len(sax_str)-self.w+1)]
        
        if not words: return counts
        
        unique_words = [words[0]]
        for i in range(1, len(words)):
            if words[i] != words[i-1]:
                unique_words.append(words[i])
                
        for w in unique_words:
            if w in self.vocab_map:
                counts[self.vocab_map[w]] += 1
        return counts


def strict_split(data_dir, test_ratio=0.2):
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return [], [], [], [], []
        
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    train_files, test_files, train_labels, test_labels = [], [], [], []
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_path, "*.csv"))) 
        if not files: continue
        
        n_test = int(len(files) * test_ratio)
        if n_test == 0 and len(files) > 1: n_test = 1
        
        t_files = files[:-n_test] if n_test > 0 else files
        v_files = files[-n_test:] if n_test > 0 else []
        
        train_files.extend(t_files)
        test_files.extend(v_files)
        train_labels.extend([cls] * len(t_files))
        test_labels.extend([cls] * len(v_files))
        
    return train_files, test_files, train_labels, test_labels, classes

if __name__ == '__main__':
    train_files, test_files, y_train, y_test, classes = strict_split(DATA_DIR)
    
    if not train_files:
        exit()

    print(f"training: {len(train_files)}, testing: {len(test_files)}")
    print(f"PAA={PAA_WINDOW}, Alphabet={ALPHABET_SIZE}, WordLen={WORD_LENGTH}")

    extractor = SaxBopExtractor(ALPHABET_SIZE, WORD_LENGTH, PAA_WINDOW)

    def extract_dataset(files, labels):
        X_list = []
        y_list = []
        for i, (fpath, label) in enumerate(zip(files, labels)):
            try:
                df = pd.read_csv(fpath)
                sample_feats = []
                for col, need_diff in COLUMN_CONFIG.items():

                    if col not in df.columns:
                        data = np.zeros(SEQUENCE_LENGTH)
                    else:
                        data = df[col].values.astype(float)
                    
                    if need_diff: 
                        data = np.diff(data)
                    
                    if len(data) > SEQUENCE_LENGTH:
                        data = data[:SEQUENCE_LENGTH]
                    
                    feat = extractor.transform_channel(data)
                    sample_feats.append(feat)
                
                X_list.append(np.hstack(sample_feats))
                y_list.append(label)
                
                if (i+1) % 100 == 0:
                    print(f" {i+1} samples...", end='\r')
            except Exception as e:
                pass
        return np.array(X_list), np.array(y_list)

    X_train, y_train = extract_dataset(train_files, y_train)

    X_test, y_test = extract_dataset(test_files, y_test)
    

    print(" (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [1, 10, 100],           
        'gamma': ['scale', 'auto'],   
        'kernel': ['rbf']
    }
    
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=1, verbose=2)
    grid.fit(X_train_scaled, y_train)
    
    print(f"{grid.best_params_}")
    best_clf = grid.best_estimator_

    y_pred = best_clf.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))