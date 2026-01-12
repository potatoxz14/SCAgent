import numpy as np
import joblib
from cuml.decomposition import PCA as cuML_PCA

import cupy as cp 
from typing import Union, Tuple

def train_and_save_pca(
    X: np.ndarray, 
    n_components: int, 
    model_path: str = 'pca_model.pkl'
) -> Tuple[np.ndarray, cuML_PCA]:
    print(f"original: {X.shape[1]}")
    print(f"target: {n_components}")
    
    X_gpu = cp.asarray(X)
    
    pca = cuML_PCA(n_components=n_components)
    
    X_reduced_gpu = pca.fit_transform(X_gpu)
    
    X_reduced = cp.asnumpy(X_reduced_gpu)
    
    joblib.dump(pca, model_path)
    
    print(f"{X_reduced.shape[1]}")
    print(f"{model_path}")
    
    return X_reduced, pca

def load_and_transform_pca(
    X_new: np.ndarray, 
    model_path: str = 'pca_model.pkl'
) -> np.ndarray:

    print(f"loading: {model_path}")
    
    try:
        pca = joblib.load(model_path)
    except FileNotFoundError:
        return np.array([])
    
    X_new_gpu = cp.asarray(X_new)
    
    X_reduced_new_gpu = pca.transform(X_new_gpu)
    
    X_reduced_new = cp.asnumpy(X_reduced_new_gpu)
    
    return X_reduced_new
