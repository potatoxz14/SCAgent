import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import optuna

DATA_DIR = "./data_app_all_v2_with_diff"
SEQUENCE_LENGTH = 1000
N_TRIALS = 20  
EPOCHS_PER_TRIAL = 15  


COLUMN_CONFIG = {
    'en0_obytes': False, 'en0_ibytes': False, 'mem_cow_faults': False, 'mem_wire_cnt': False,
    'fs_delta_bytes': False, 'ane_lat_ms': False, 'gpu_lat_ms': False, 'jitter_us': False,
}
SELECTED_COLUMNS = list(COLUMN_CONFIG.keys())


class MischiefDataset(Dataset):
    def __init__(self, file_paths, labels, sequence_length=500):
        self.file_paths = file_paths
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            df = pd.read_csv(path)
            processed_channels = []
            for col, need_diff in COLUMN_CONFIG.items():
                if col not in df.columns: col_data = np.zeros(len(df))
                else: col_data = df[col].values.astype(float)
                if need_diff:
                    col_data = np.insert(np.diff(col_data), 0, 0)
                processed_channels.append(col_data)
            data = np.stack(processed_channels, axis=1)
            if len(data) > self.sequence_length: data = data[:self.sequence_length, :]
            else: data = np.vstack((data, np.zeros((self.sequence_length - len(data), data.shape[1]))))
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0 
            data = (data - mean) / std
            return torch.FloatTensor(data).transpose(0, 1), torch.tensor(label, dtype=torch.long)
        except: return torch.zeros((len(SELECTED_COLUMNS), self.sequence_length)), torch.tensor(label, dtype=torch.long)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1) if self.use_bottleneck else nn.Identity()
        input_channels = bottleneck_channels if self.use_bottleneck else in_channels
        self.conv_layers = nn.ModuleList([nn.Conv1d(input_channels, out_channels, k, padding=k//2) for k in kernel_sizes])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x_bottleneck = self.bottleneck(input_tensor)
        outputs = [layer(x_bottleneck) for layer in self.conv_layers]
        outputs.append(self.conv_pool(self.maxpool(input_tensor)))
        return self.act(self.bn(torch.cat(outputs, dim=1)))

class InceptionTime(nn.Module):
    def __init__(self, num_channels, num_classes, num_blocks=3, num_filters=32):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            in_c = num_channels if i == 0 else num_filters * 4
            layers.append(InceptionModule(in_c, num_filters))
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        return self.fc(x.squeeze(-1))

def strict_split(data_dir, test_ratio=0.2):
    if not os.path.exists(data_dir): return [], [], [], [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    train_files, test_files, train_labels, test_labels = [], [], [], []
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_path, "*.csv")))
        if not files: continue
        num_test = int(len(files) * test_ratio)
        if num_test == 0 and len(files) > 1: num_test = 1
        train_files.extend(files[:-num_test] if num_test > 0 else files)
        test_files.extend(files[-num_test:] if num_test > 0 else [])
        train_labels.extend([cls] * (len(files)-num_test))
        test_labels.extend([cls] * num_test)
    return train_files, test_files, train_labels, test_labels, classes


def objective(trial):

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_blocks = trial.suggest_int("num_blocks", 1, 4)      
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64]) 


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    model = InceptionTime(
        num_channels=len(SELECTED_COLUMNS), 
        num_classes=len(classes),
        num_blocks=num_blocks,
        num_filters=num_filters
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc_this_trial = 0.0
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        best_acc_this_trial = max(best_acc_this_trial, acc)

        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_acc_this_trial

if __name__ == '__main__':
    X_train, X_test, y_train_raw, y_test_raw, classes = strict_split(DATA_DIR, test_ratio=0.2)
    if not X_train: exit()
    
    le = LabelEncoder()
    le.fit(classes)
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    train_ds = MischiefDataset(X_train, y_train, SEQUENCE_LENGTH)
    test_ds = MischiefDataset(X_test, y_test, SEQUENCE_LENGTH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}")
    print(f" Optuna  ({N_TRIALS} try)...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n================ ================")
    print(f"best accuracy: {study.best_value:.4f}")
    print("optimal subset:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(" (Epochs=50)...")
    best_params = study.best_params
    
    final_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)
    
    final_model = InceptionTime(
        num_channels=len(SELECTED_COLUMNS), 
        num_classes=len(classes),
        num_blocks=best_params['num_blocks'],
        num_filters=best_params['num_filters']
    ).to(device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()
    
    final_best_acc = 0
    for epoch in range(50):
        final_model.train()
        for X, y in final_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = final_model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        final_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = final_model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > final_best_acc: final_best_acc = acc
        print(f"Final Epoch {epoch+1}: Acc: {acc:.4f}")
        
    print(f"final best accuracy: {final_best_acc:.4%}")