import optuna
import torch.optim as optim #[cite: 1]

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Inizializza i dataloader e i modelli (qui importati dai file creati precedentemente)
    # train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    # model = SimpleCNN(num_classes=5)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Esegui train_one_epoch in un loop 
    # e restituisci l'accuratezza di validazione
    # return val_accuracy

# Per lanciare:
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)