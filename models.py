import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_classes=7):
        super(BaselineModel, self).__init__()
        
        # Path 1: MaxPool (2x2) Stride 2[cite: 1]
        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Path 2: Conv 5@(2x2) S2-ReLu[cite: 1]
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # Path 3: Tre convoluzioni in cascata[cite: 1]
        self.branch3 = nn.Sequential(
            # Conv 3@(1x1) S1-ReLu[cite: 1]
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU(),
            # Conv 6@(2x2) S1-ReLu (usiamo padding='same' per mantenere le dimensioni per la successiva)[cite: 1]
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding='same'),
            nn.ReLU(),
            # Conv 9@(4x4) S2-ReLu (padding 1 per allineare l'output a 170x50)[cite: 1]
            nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Post-Concat: Conv 3@(1x1) S1-ReLu[cite: 1]
        # I canali in input saranno la somma dei canali dei 3 branch: 1 + 5 + 9 = 15
        self.post_concat = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU()
        )
        
        # Classifier: Flatten -> Dropout (0.2) -> Dense[cite: 1]
        # Dimensione dopo le feature maps: 3 canali * 170 * 50 = 25500
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(25500, num_classes)
        )

    def forward(self, x):
        # x shape attesa: (batch_size, 1, 340, 100)[cite: 1]
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatena lungo la dimensione dei canali[cite: 1]
        out = torch.cat([out1, out2, out3], dim=1)
        
        out = self.post_concat(out)
        out = self.classifier(out)
        
        return out