import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SHARPDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        self.data = []
        self.labels = []
        
        # Mappatura delle etichette (5 classi base)
        # E=Empty, L=Sitting, W=Walking, R=Running, J=Jumping
        self.label_map = {'E': 0, 'L': 1, 'W': 2, 'R': 3, 'J': 4} 
        
        # Training prende solo le cartelle che iniziano per S1, Test prende tutte le altre
        target_scenarios = ['S1'] if mode == 'train' else ['S2', 'S3', 'S4', 'S5', 'S6', 'S7']
        
        print(f"Caricamento dati per la fase di: {mode.upper()}...")
        
        # Esplora le cartelle (S1a, S1b, S2a, ecc.)
        for folder in os.listdir(data_path):
            if not any(folder.startswith(s) for s in target_scenarios):
                continue
                
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            # Trova tutte le attività uniche nella cartella (E, L, W, R, J)
            files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            activities = set([f.split('_')[1] for f in files])
            
            for act in activities:
                if act not in self.label_map:
                    continue # Ignora attività extra (S, C, G) per ora
                
                antennas_data = []
                # Carica i file delle 4 antenne (stream_0, stream_1, stream_2, stream_3)
                for stream_idx in range(4):
                    file_name = f"{folder}_{act}_stream_{stream_idx}.txt"
                    file_path = os.path.join(folder_path, file_name)
                    
                    if os.path.exists(file_path):
                        # Il trucco è qui: np.load legge il file "finto" .txt come matrice numpy
                        stream_array = np.load(file_path, allow_pickle=True)
                        antennas_data.append(stream_array)
                        
                # Se abbiamo trovato tutte e 4 le antenne, salviamo il campione
                if len(antennas_data) == 4:
                    stacked_data = np.stack(antennas_data) # Shape: (4, 340, 100)
                    self.data.append(stacked_data)
                    self.labels.append(self.label_map[act])
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(f"[{mode.upper()}] Caricati {len(self.labels)} campioni. Shape dati: {self.data.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32) 
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.mode == 'train':
            # Training: estraiamo un'antenna a caso (Data Augmentation temporale/spaziale)
            antenna_idx = np.random.randint(0, 4)
            x = x[antenna_idx].unsqueeze(0) # Shape: (1, 340, 100)
            return x, y
        else:
            # Test: le manteniamo tutte e 4 per la Decision Fusion
            x = x.unsqueeze(1) # Shape: (4, 1, 340, 100)
            return x, y
            
def get_dataloaders(data_dir, batch_size=32):
    train_dataset = SHARPDataset(data_dir, mode='train')
    test_dataset = SHARPDataset(data_dir, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader