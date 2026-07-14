import torch #[cite: 1, 2, 3, 4]
import torch.nn as nn #[cite: 1, 2, 3, 4]
from tqdm import tqdm #[cite: 1, 2]

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train() #[cite: 2]
    total_loss, total_acc, n = 0.0, 0.0, 0 #[cite: 2]
    
    for inputs, labels in tqdm(dataloader): #[cite: 2]
        inputs = inputs.to(device, non_blocking=True) #[cite: 2]
        labels = labels.to(device, non_blocking=True) #[cite: 2]
        
        optimizer.zero_grad() #[cite: 2]
        logits = model(inputs) #[cite: 2]
        loss = criterion(logits, labels) 
        loss.backward() #[cite: 2]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #[cite: 2]
        optimizer.step() #[cite: 2]
        
        acc = (logits.argmax(dim=-1) == labels).float().mean() #[cite: 2]
        total_loss += loss.item() #[cite: 2]
        total_acc += acc.item() #[cite: 2]
        n += 1 #[cite: 2]
        
    return total_loss / n, total_acc / n #[cite: 2]