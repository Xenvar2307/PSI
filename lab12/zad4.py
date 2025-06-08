import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, patience=10, delta=0.001, mode='min', verbose=False, path='best_model.pt'):
        """
        Args:
            model: Model PyTorch do treningu.
            criterion: Funkcja straty (np. nn.MSELoss).
            optimizer: Optymalizator (np. optim.SGD).
            patience (int): Liczba epok bez poprawy, po której trening zostanie zatrzymany.
            delta (float): Minimalna zmiana metryki uznawana za poprawę.
            mode (str): 'min' (minimalizacja, np. strata) lub 'max' (maksymalizacja, np. dokładność).
            verbose (bool): Jeśli True, drukuje komunikaty o postępie.
            path (str): Ścieżka do zapisu najlepszego modelu.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []

    def train_step(self, X, y):
        """Wykonuje krok treningowy."""
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, X, y):
        """Oblicza stratę na zbiorze walidacyjnym."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return loss.item()

    def early_stopping(self, score, epoch):
        """Sprawdza warunek Early Stopping i zapisuje najlepszy model."""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
            self.best_epoch = epoch
        elif (self.mode == 'min' and score < self.best_score - self.delta) or \
             (self.mode == 'max' and score > self.best_score + self.delta):
            # TODO: Ustaw best_score, best_epoch i counter - DONE
            self.save_checkpoint()
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0
            if self.verbose:
                print(f'Metryka poprawiona: {self.best_score:.4f} w epoce {epoch}')
        else:
            # TODO: Zwiększ counter i sprawdź warunek - DONE
            self.counter += 1
            if self.verbose:
                print(f'Brak poprawy przez {self.counter} epok')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early Stopping: Zatrzymano trening w epoce {epoch}')

    def save_checkpoint(self):
        """Zapisuje stan modelu."""
        torch.save(self.model.state_dict(), self.path)

    def train(self, X_train, y_train, X_val, y_val, num_epochs):
        """
        Trenuje model z Early Stopping i zapisuje straty.
        Args:
            X_train, y_train: Dane treningowe.
            X_val, y_val: Dane walidacyjne.
            num_epochs (int): Maksymalna liczba epok.
        """
        for epoch in range(num_epochs):
            # Krok treningowy
            train_loss = self.train_step(X_train, y_train)
            self.train_losses.append(train_loss)
            
            # Walidacja
            val_loss = self.validate(X_val, y_val)
            self.val_losses.append(val_loss)

            # logowanie metryk
            wandb.log({"train_loss":train_loss, "val_loss": val_loss}, step = epoch)
            
            # Early Stopping
            self.early_stopping(val_loss, epoch + 1)
            if self.early_stop:
                break
                raise NotImplementedError("Dokończ kod!")

    def plot_losses(self):
        """Generuje wykres strat z punktem Early Stopping."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Strata treningowa', color='blue')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Strata walidacyjna', color='orange')
        if self.early_stop:
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', label='Early Stopping')
            plt.scatter(self.best_epoch, self.val_losses[self.best_epoch - 1], color='red', s=100, zorder=5)
        plt.title('Wizualizacja Early Stopping')
        plt.xlabel('Epoki')
        plt.ylabel('Strata (MSE)')
        plt.legend()
        plt.grid(True)

# start code
import wandb

wandb.login(key = "cbc01cde6ab951d7dc84439bfb25ef2f5425b383")

wandb.init(project="mnist",
           config= {
               "learning_rate": 0.001, "epochs": 100,
               "batch_size": 128, "dropout": 0.5
           })

# Dane MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Spłaszczenie obrazów (28x28 -> 784)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)

# Przygotowanie danych (spłaszczone obrazy)
X_train, y_train = next(iter(train_loader))
X_train = X_train.view(-1, 28 * 28)  # Spłaszczenie: (batch_size, 1, 28, 28) -> (batch_size, 784)
y_train = y_train.long()
X_val, y_val = next(iter(val_loader))
X_val = X_val.view(-1, 28 * 28)  # Spłaszczenie
y_val = y_val.long()

# Definicja modelu
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(wandb.config['dropout']),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(wandb.config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(wandb.config['dropout']),
            nn.Linear(128, 10)  # 10 klas dla MNIST
        )
    
    def forward(self, x):
        return self.network(x)

# Inicjalizacja i trening
torch.manual_seed(42)
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), wandb.config['learning_rate'])

trainer = ModelTrainer(model, criterion, optimizer, patience=10, delta=0.001, mode='min', verbose=False, path='best_mnist_model.pt')
trainer.train(X_train, y_train, X_val, y_val, num_epochs=wandb.config['epochs'])

