import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import itertools


DATA_DIR = "./my_collected_data"  
SEQUENCE_LENGTH = 5000           


PAA_WINDOW_SIZE = 5      
ALPHABET_SIZE = 5        
WORD_LENGTH = 5          

TARGET_COLUMNS = [
    'free_count', 'active_count', 'zero_fill_count', 'faults', # VM
    'en0_ibytes', 'en0_obytes'                                 # Network
]


class SaxBopTransformer:
    def __init__(self, alphabet_size, word_length, paa_window):
        self.a = alphabet_size
        self.w = word_length
        self.p = paa_window

        self.breakpoints = stats.norm.ppf(np.linspace(1./self.a, 1-1./self.a, self.a-1))
        
        chars = [chr(97 + i) for i in range(self.a)] # 'a', 'b', ...
        self.vocab = [''.join(p) for p in itertools.product(chars, repeat=self.w)]
        self.vocab_map = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def z_normalize(self, series):
        """ Z-normalization  """
        std = np.std(series)
        if std == 0: return np.zeros_like(series)
        return (series - np.mean(series)) / std

    def apply_paa(self, series):
        """ Piecewise Aggregate Approximation  """

        n_segments = len(series) // self.p
        reshaped = series[:n_segments*self.p].reshape(n_segments, self.p)
        return reshaped.mean(axis=1)

    def discretize(self, paa_series):

        indices = np.digitize(paa_series, self.breakpoints)
        return "".join([chr(97 + i) for i in indices])

    def bag_of_patterns(self, sax_string):
        feature_vector = np.zeros(self.vocab_size, dtype=int)
        
        words = []
        for i in range(len(sax_string) - self.w + 1):
            words.append(sax_string[i : i + self.w])
        
        if not words:
            return feature_vector
            
        unique_consecutive_words = [words[0]]
        for i in range(1, len(words)):
            if words[i] != words[i-1]:
                unique_consecutive_words.append(words[i])

        for word in unique_consecutive_words:
            if word in self.vocab_map:
                feature_vector[self.vocab_map[word]] += 1
                
        return feature_vector

    def transform(self, raw_series):
        # 1. Z-Norm
        norm_series = self.z_normalize(raw_series)
        # 2. PAA
        paa_series = self.apply_paa(norm_series)
        # 3. Discretization (SAX String)
        sax_string = self.discretize(paa_series)
        # 4. BOP Feature Vector
        return self.bag_of_patterns(sax_string)

def load_and_extract_features():
    X_features = []
    y_labels = []
    
    print(f"BOP: {transformer.vocab_size}")
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"classes: {classes}")

    for cls_idx, cls_name in enumerate(classes):
        cls_path = os.path.join(DATA_DIR, cls_name)
        files = glob.glob(os.path.join(cls_path, "*.csv"))
        
        print(f" {cls_name}, : {len(files)}...")
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                cols_to_use = [c for c in TARGET_COLUMNS if c in df.columns]
                if not cols_to_use: continue
                
                data = df[cols_to_use].values
                
                if len(data) > SEQUENCE_LENGTH:
                    data = data[:SEQUENCE_LENGTH]
                else:
                    padding = np.zeros((SEQUENCE_LENGTH - len(data), data.shape[1]))
                    data = np.vstack((data, padding))
                
                data_diff = np.diff(data, axis=0) 
                
                sample_features = []
                for col_idx in range(data_diff.shape[1]):
                    channel_series = data_diff[:, col_idx]

                    bop_vec = transformer.transform(channel_series)
                    sample_features.append(bop_vec)
                
                # Concatenate these sequences into one BOP array
                final_feature_vector = np.hstack(sample_features)
                
                X_features.append(final_feature_vector)
                y_labels.append(cls_idx) 
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X_features), np.array(y_labels)

if __name__ == "__main__":
    X, y = load_and_extract_features()
    
    if len(X) == 0:
        exit()
        
    print(f"samples: {X.shape[0]}, features: {X.shape[1]}")
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    

    clf = SVC(kernel='rbf', probability=True, gamma='scale')
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Top-1 : {acc*100:.2f}%")