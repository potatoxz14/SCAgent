import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


DATA_DIR = "./data_web_all_v1" 
SEQUENCE_LENGTH = 640        
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3


CHANNEL_CONFIG = {
    'net_en0_obytes': True,
    'net_en0_ibytes': True,
    'net_en0_opackets': True,
    'mem_wired': True,
    'mem_zero_fill': True,
    'fs_free_bytes': True,     
    'latency_cache': False,
    'latency_metal': False,
    'latency_filesystem': False,
    'latency_coreml': False
}
SELECTED_KEYS = list(CHANNEL_CONFIG.keys())

class WebsiteFingerprintingDataset(Dataset):
    def __init__(self, file_paths, labels, sequence_length=500):
        self.file_paths = file_paths
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            df = pd.read_csv(path)
            processed_channels = []
            
            for col_name, need_diff in CHANNEL_CONFIG.items():

                actual_col = col_name
                if col_name == 'fs_free_mem' and 'fs_free_bytes' in df.columns:
                    actual_col = 'fs_free_bytes'

                if actual_col not in df.columns:
                    raw_data = np.zeros(len(df))
                else:
                    raw_data = df[actual_col].values.astype(float)
                
                if need_diff:
                    diff = np.diff(raw_data)
                    data_col = np.insert(diff, 0, 0)
                else:
                    data_col = raw_data
                
                processed_channels.append(data_col)
            
            data = np.stack(processed_channels, axis=1)
            
            if len(data) > self.sequence_length:
                data = data[:self.sequence_length, :]
            else:
                padding = np.zeros((self.sequence_length - len(data), data.shape[1]))
                data = np.vstack((data, padding))
            
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0
            data = (data - mean) / std

            data_tensor = torch.FloatTensor(data).transpose(0, 1)
            return data_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return torch.zeros((len(SELECTED_KEYS), self.sequence_length)), torch.tensor(label, dtype=torch.long)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1) if self.use_bottleneck else nn.Identity()
        input_channels = bottleneck_channels if self.use_bottleneck else in_channels
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_channels, out_channels, k, padding=k//2) for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x_bottle = self.bottleneck(input_tensor)
        outputs = [layer(x_bottle) for layer in self.conv_layers]
        outputs.append(self.conv_pool(self.maxpool(input_tensor)))
        return self.act(self.bn(torch.cat(outputs, dim=1)))

class InceptionTime(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionModule(num_channels, 32),
            InceptionModule(32 * 4, 32),
            InceptionModule(32 * 4, 32)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def strict_split(data_dir, test_ratio=0.2):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    train_files, test_files = [], []
    train_labels, test_labels = [], []
    
    print(f"  {test_ratio})...")
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_path, "*.csv")))
        
        if not files:
            continue
            
        num_samples = len(files)
        num_test = int(num_samples * test_ratio)
        if num_test == 0 and num_samples > 1:
            num_test = 1 
        
        t_files = files[:-num_test] if num_test > 0 else files
        v_files = files[-num_test:] if num_test > 0 else []
        
        train_files.extend(t_files)
        test_files.extend(v_files)
        train_labels.extend([cls] * len(t_files))
        test_labels.extend([cls] * len(v_files))
        
    print(f" training: {len(train_files)} , testing: {len(test_files)} 个")
    return train_files, test_files, train_labels, test_labels, classes


if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        print(f"error: {DATA_DIR} does not exist")
        exit()

    X_train, X_test, y_train_raw, y_test_raw, classes = strict_split(DATA_DIR, test_ratio=0.2)

    if len(X_train) == 0:
        exit()

    le = LabelEncoder()
    le.fit(classes) 
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # Loader
    train_ds = WebsiteFingerprintingDataset(X_train, y_train, SEQUENCE_LENGTH)
    test_ds = WebsiteFingerprintingDataset(X_test, y_test, SEQUENCE_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}")
    
    model = InceptionTime(num_channels=len(SELECTED_KEYS), num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        loss_sum, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
        train_acc = 100 * correct / total
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_t, y_t in test_loader:
                X_t, y_t = X_t.to(device), y_t.to(device)
                out_t = model(X_t)
                test_correct += (out_t.argmax(1) == y_t).sum().item()
                test_total += y_t.size(0)
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch {epoch+1} | Loss: {loss_sum/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc

    print(f"best accuracy: {best_acc:.2f}%")