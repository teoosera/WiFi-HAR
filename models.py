import torch #[cite: 1, 2, 3, 4]
import torch.nn as nn #[cite: 1, 2, 3, 4]
import math #[cite: 2]

# --- MODELLO 1: Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__() #
        # Utilizziamo Sequential per concatenare facilmente i layer[cite: 3]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), #[cite: 3]
            nn.ReLU(inplace=True), #[cite: 3]
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) #[cite: 3]
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 48 * 48, out_features=64), #[cite: 3]
            nn.ReLU(True), #[cite: 3]
            nn.Linear(in_features=64, out_features=num_classes) #[cite: 3]
        )

    def forward(self, x):
        x = self.encoder_cnn(x) #[cite: 3]
        x = self.flatten(x) #[cite: 3]
        x = self.classifier(x) #[cite: 3]
        return x #[cite: 3]

# --- MODELLO 2: CNN con Inception e ResNet ---
class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Le architetture Inception utilizzano rami multipli in parallelo[cite: 4]
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )
        self.combine = nn.Conv2d(32, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.combine(out)
        # La connessione residua facilita l'ottimizzazione e il training[cite: 4]
        return self.relu(x + out)

class InceptionResNetCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.block = InceptionResNetBlock(32)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(32 * 48 * 48, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.block(x)
        x = self.flatten(x)
        return self.fc(x)

# --- MODELLO 3: CNN Encoder + Transform Head ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__() #[cite: 2]
        self.pos_encoding = torch.zeros(max_len, d_model)
        # I vettori sinusoidali permettono al modello di distinguere l'ordine dei token[cite: 2]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[cite: 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #[cite: 2]
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1) #[cite: 2]
        return x + self.pos_encoding[:, :seq_len, :].to(x.device) #[cite: 2]

class CNNTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, num_heads=4):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1), #[cite: 3]
            nn.ReLU(inplace=True), #[cite: 3]
            nn.Flatten(start_dim=2) 
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        # Operazione di attenzione basata su nn.MultiheadAttention[cite: 2]
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True) #[cite: 2]
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        features = self.encoder_cnn(x) 
        features = features.permute(0, 2, 1) 
        x = self.pos_encoder(features)
        attn_output, _ = self.attention(query=x, key=x, value=x) #[cite: 2]
        pooled_output = attn_output.mean(dim=1)
        return self.fc(pooled_output)