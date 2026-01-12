import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_DIR = "./my_collected_data"  
BATCH_SIZE = 16                   
EPOCHS = 50                      
LR = 1e-3                        

TARGET_COLUMNS = [
    'en0_ibytes', 
    'en0_obytes', 
    'user_time',
    'cow_faults'
] 


class SideChannelDataset(Dataset):
    def __init__(self, file_paths, labels, selected_columns=None, sequence_length=5000):

        self.file_paths = file_paths
        self.labels = labels
        self.selected_columns = selected_columns
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            df = pd.read_csv(path)

            if self.selected_columns:

                available_cols = [c for c in self.selected_columns if c in df.columns]
                # if len(available_cols) < len(self.selected_columns):
                #     print(f" {path} : {available_cols}")
                
                df = df[available_cols]
            
            data = df.values 
            
            if len(data) > self.sequence_length:
                data = data[:self.sequence_length, :]
            else:
                padding = np.zeros((self.sequence_length - len(data), data.shape[1]))
                data = np.vstack((data, padding))
                

            data_diff = np.diff(data, axis=0) 

            data_diff = np.vstack((np.zeros((1, data.shape[1])), data_diff))

            data_tensor = torch.FloatTensor(data_diff).transpose(0, 1)
            
            return data_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            num_features = len(self.selected_columns) if self.selected_columns else 1
            return torch.zeros((num_features, self.sequence_length)), torch.tensor(label, dtype=torch.long)
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            input_channels = bottleneck_channels
        else:
            input_channels = in_channels
            
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_layers.append(
                nn.Conv1d(input_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        if self.use_bottleneck:
            input_tensor = self.bottleneck(input_tensor)
        outputs = [layer(input_tensor) for layer in self.conv_layers]
        pool_out = self.maxpool(x)
        pool_out = self.conv_pool(pool_out)
        outputs.append(pool_out)
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        return self.act(x)

class InceptionTime(nn.Module):
    #  Inception Blocks [cite: 342]
    def __init__(self, num_channels, num_classes):
        super(InceptionTime, self).__init__()
        self.blocks = nn.Sequential(
            InceptionModule(num_channels, 32),
            InceptionModule(32 * 4, 32),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':

    all_files = []
    all_labels = []
    
    if not os.path.exists(DATA_DIR):
        print(f"error: dir {DATA_DIR} does not exist")
        exit()

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f" {len(classes)} (web /APP): {classes[:5]}...")

    le = LabelEncoder()
    
    for cls_name in classes:
        cls_path = os.path.join(DATA_DIR, cls_name)
        files = glob.glob(os.path.join(cls_path, "*.csv"))
        all_files.extend(files)
        all_labels.extend([cls_name] * len(files))

    print(f" {len(all_files)} files")
    
    if len(all_files) == 0:
        exit()

    all_labels_enc = le.fit_transform(all_labels)
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, all_labels_enc, test_size=0.2, stratify=all_labels_enc, random_state=42
    )

    train_ds = SideChannelDataset(X_train, y_train, selected_columns=TARGET_COLUMNS)
    test_ds = SideChannelDataset(X_test, y_test, selected_columns=TARGET_COLUMNS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    sample_data, _ = train_ds[0]
    num_input_channels = sample_data.shape[0] 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {device}")

    model = InceptionTime(num_channels=num_input_channels, num_classes=num_classes).to(device)
    

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    print(f"accuracy: {100. * test_correct / test_total:.2f}%")