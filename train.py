import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # IMPORTANTE: Aggiunto per TensorBoard
from models import BaselineModel
from dataset import get_dataloaders

# Aggiunto log_dir nei parametri
def train(data_dir, checkpoint_dir, log_dir, epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inizio training su: {device}")
    
    # Inizializza il writer di TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    model = BaselineModel(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ---------------- GESTIONE CHECKPOINT ----------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Trovato checkpoint in {checkpoint_path}. Caricamento in corso...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Allenamento ripreso dall'epoca {start_epoch+1}")
    # -----------------------------------------------------

    train_loader = get_dataloaders(data_dir, batch_size)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        loss_epoch = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss_epoch:.4f} | Accuracy: {acc:.2f}%")
        
        # -----> SCRITTURA SU TENSORBOARD <-----
        writer.add_scalar('Training/Loss', loss_epoch, epoch)
        writer.add_scalar('Training/Accuracy', acc, epoch)
        
        # Salvataggio Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_epoch,
        }, checkpoint_path)

    # Chiudi il writer alla fine
    writer.close()

if __name__ == "__main__":
    # PERCORSI SUL TUO DRIVE (Codice preso da Github, ma dati e log salvati su Drive)
    DATA_DIR = "/content/drive/MyDrive/WiFi-HAR/doppler_traces" 
    CHECKPOINT_DIR = "/content/drive/MyDrive/WiFi-HAR/weights"
    LOG_DIR = "/content/drive/MyDrive/WiFi-HAR/logs" # Cartella per Tensorboard
    
    train(DATA_DIR, CHECKPOINT_DIR, LOG_DIR, epochs=5)

