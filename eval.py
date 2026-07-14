import os
import torch
from models import BaselineModel
from dataset import get_dataloaders

def evaluate(data_dir, checkpoint_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BaselineModel(num_classes=7).to(device)
    
    # Caricamento del dizionario dal checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Estraiamo solo i pesi della rete ('model_state_dict')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modello caricato dal checkpoint (Epoca completata: {checkpoint['epoch']+1})")
    else:
        raise FileNotFoundError(f"Checkpoint non trovato nel percorso: {checkpoint_path}")
        
    model.eval()
    
    test_loader = get_dataloaders(data_dir, batch_size)
    
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print(f"Performance sul Test Set: {acc:.2f}%")

if __name__ == "__main__":
    # Sostituisci con i percorsi corretti di Drive
    TEST_DATA_DIR = "/content/drive/MyDrive/Cartella_Progetto_WiFi/dataset_test"
    CHECKPOINT_PATH = "/content/drive/MyDrive/Cartella_Progetto_WiFi/checkpoints/checkpoint.pth"
    
    evaluate(TEST_DATA_DIR, CHECKPOINT_PATH)