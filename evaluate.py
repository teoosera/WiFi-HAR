import torch #[cite: 1, 2, 3, 4]
import matplotlib.pyplot as plt #[cite: 1, 2, 3, 4]
import numpy as np #[cite: 1, 2, 3, 4]
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

@torch.no_grad() #[cite: 2]
def evaluate(model, loader, device):
    model.eval() #[cite: 2]
    all_preds, all_labels = [], []
    
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True) #[cite: 2]
        logits = model(inputs) #[cite: 2]
        preds = logits.argmax(dim=-1) #[cite: 2]
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    print(classification_report(all_labels, all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis') #[cite: 1]
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()