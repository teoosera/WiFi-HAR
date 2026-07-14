import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WiFiActivityDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        
        # Mappa provvisoria per estrarre l'etichetta.
        # Dovrai adattare le chiavi in base a come sono salvati esattamente i nomi.
        self.activity_map = {
            'E': 0, # Empty[cite: 1]
            'S': 1, # Sitting[cite: 1]
            'W': 2, # Walking[cite: 1]
            'R': 3, # Running[cite: 1]
            'J': 4, # Jumping[cite: 1]
            'A': 5, # Sesto placeholder
            'B': 6  # Settimo placeholder
        }
        self._load_dataset()

    def _load_dataset(self):
        # Esplora le cartelle come S1a_J1
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            if os.path.isdir(folder_path):
                # Estrae il carattere dell'attività (Esempio base: 'J' da 'S1a_J1')
                try:
                    activity_code = folder_name.split('_')[1][0] 
                except IndexError:
                    continue
                
                label = self.activity_map.get(activity_code.upper(), -1)
                
                if label != -1:
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.txt'):
                            self.samples.append(os.path.join(folder_path, file_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        
        # Carica il file txt numpy (forma attesa: 340x100)[cite: 1]
        data = np.loadtxt(file_path)
        
        # Aggiunge la dimensione del canale richiesta dalla CNN (1, 340, 100)[cite: 1]
        data = np.expand_dims(data, axis=0)
        
        tensor_data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return tensor_data, label

def get_dataloaders(data_dir, batch_size=32):
    dataset = WiFiActivityDataset(data_dir)
    # Ritorna il dataloader per l'addestramento
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)