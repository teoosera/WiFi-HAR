import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MODELLO 1: CNN Custom Semplice ---
class CustomSimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Da 340x100 a 170x50
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Da 170x50 a 85x25
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 85 * 25, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.classifier(x)


# --- MODELLO 2: CNN con Residual Connections (Stile ResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual # Connessione residua
        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # Da 340x100 a 170x50
            nn.ReLU(inplace=True)
        )
        self.res_block1 = ResidualBlock(16)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Da 170x50 a 85x25
            nn.ReLU(inplace=True)
        )
        self.res_block2 = ResidualBlock(32)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 85 * 25, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.downsample(x)
        x = self.res_block2(x)
        x = self.flatten(x)
        return self.fc(x)


# --- MODELLO 3: CNN ibrida con GRU (Per sequenze temporali) ---
# Usa la CNN per estrarre le feature spaziali e una RNN per le dinamiche temporali
class CNN_GRU(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)) # Riduciamo solo i bin di velocità (da 100 a 50), teniamo i 340 istanti intatti
        )
        
        # RNN layer: input size = 16 canali * 50 bin = 800
        self.rnn = nn.GRU(input_size=800, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x) # Shape: (batch, 16, 340, 50)
        
        # Riorganizziamo per la RNN: (batch, sequenza_temporale, feature)
        # Permutiamo in (batch, 340, 16, 50) e poi appiattiamo le ultime due dimensioni
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, 340, -1) # Shape: (batch, 340, 800)
        
        # Passaggio nella GRU
        out, _ = self.rnn(x) # out shape: (batch, 340, 64)
        
        # Prendiamo solo l'ultimo istante temporale per la classificazione
        last_time_step = out[:, -1, :] 
        return self.fc(last_time_step)