import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SHARPDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        # Assicurati che i file numpy si chiamino così nel tuo Drive
        self.data = np.load(f"{data_path}/{mode}_doppler.npy") 
        self.labels = np.load(f"{data_path}/{mode}_labels.npy")

        print(f"Controllo shape {mode} data: {self.data.shape}")
        print(f"Controllo shape {mode} labels: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Il dataset contiene 4 antenne, 340 istanti temporali e 100 bin di velocità
        x = torch.tensor(self.data[idx], dtype=torch.float32) 
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.mode == 'train':
            # Durante il training, estraiamo un'antenna a caso per rendere il modello invariante
            antenna_idx = np.random.randint(0, 4)
            x = x[antenna_idx].unsqueeze(0) # Shape: (1, 340, 100)
            return x, y
        else:
            # Durante il test, le manteniamo tutte e 4 per la Decision Fusion[cite: 13]
            x = x.unsqueeze(1) # Shape: (4, 1, 340, 100)
            return x, y
            
def get_dataloaders(data_dir, batch_size=32):
    train_dataset = SHARPDataset(data_dir, mode='train')
    test_dataset = SHARPDataset(data_dir, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader