import torch #[cite: 1, 2, 3, 4]
from torch.utils.data import Dataset, DataLoader #[cite: 1, 3, 4]
from torchvision import transforms #[cite: 1, 3, 4]
import numpy as np #[cite: 1, 2, 3, 4]

class WiFiHARDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Caricamento matrici CSI (Channel State Information)
        self.x = np.load(f"{data_path}_x.npy")
        self.y = np.load(f"{data_path}_y.npy")
        self.transform = transform

    def __len__(self):
        return len(self.x) #[cite: 4]

    def __getitem__(self, idx):
        x_sample = self.x[idx] #[cite: 4]
        y_sample = int(self.y[idx]) #[cite: 4]
        
        if self.transform: #[cite: 4]
            x_sample = self.transform(x_sample) #[cite: 4]
            
        return x_sample, y_sample #[cite: 4]

def get_dataloaders(data_dir, batch_size=32):
    # Data augmentation con rotazioni e flip casuali[cite: 4]
    # e Gaussian blur per perturbare i dati di training
    train_transform = transforms.Compose([
        transforms.ToTensor(), #[cite: 1, 3, 4]
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), #[cite: 4]
        transforms.RandomHorizontalFlip(p=0.5), #[cite: 1, 4]
        transforms.GaussianBlur(kernel_size=3) #[cite: 1]
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor() #[cite: 1, 3, 4]
    ])

    train_dataset = WiFiHARDataset(f"{data_dir}/train", transform=train_transform)
    test_dataset = WiFiHARDataset(f"{data_dir}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #[cite: 1, 3, 4]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #[cite: 1, 3, 4]

    return train_loader, test_loader