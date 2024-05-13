import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

# for aesthetics/plotting
sns.set_theme()
plt.style.use("seaborn-v0_8")

class AutoencoderAnomalyDetection(nn.Module):
    def __init__(self, shape=[3, 32, 700], latent_dim=128):
        super().__init__()
        self.shape = shape
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(shape), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = self.encoder = nn.Sequential(
            nn.Flatten(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(shape)),
            nn.Unflatten(1, shape)
        )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)



def train_loop(dataloader, model, loss_func, optimizer, batch_size):
    model.train()
    train_loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        train_loss += loss_func(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Backpropogation.
        loss.backwards()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        train_loss /= num_batches
        correct /= size 
        
    return train_loss, correct

    
def test_loop(dataloader, model, loss_func):
    model.eval()
    test_loss, correct = 0, 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with torch.no_grad(): # No gradient calculated.
        for (X, y) in dataloader:
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct


def train_model(dataloader, model, loss_func, optimizer, epochs, batch_size=32,
                graph_transparency=True, graph_path=None):
    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t], train_accuracy[t] = train_loop(
            dataloader, model, loss_func, optimizer, batch_size=32
        )
        test_loss[t], test_accuracy[t] = test_loop(
            dataloader, model, loss_func
        )
        print("Done!")

        visualization(train_loss, train_accuracy, test_loss, test_accuracy,
                      graph_transparency, graph_path)

    return train_loss, train_accuracy, test_loss, test_accuracy

def visualization(train_loss, train_accuracy, test_loss, test_accuracy,
                  graph_transparency, graph_path):
    if graph_transparency:
        text_color = 'white'
        face_color = 'none'
    else:
        text_color = 'black'
        face_color = 'white'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=face_color)
    
    # Plot training and validation loss on the first subplot
    ax1.plot(train_loss, label="Training loss")
    ax1.plot(test_loss, label="Validation loss")
    ax1.set_title("Loss Progression", color=text_color)
    ax1.set_xlabel("Epoch number", color=text_color)
    ax1.set_ylabel("Loss", color=text_color)
    ax1.legend(facecolor=face_color)
    ax1.tick_params(axis='both', colors=text_color)

    # Plot training and validation accuracy on the second subplot
    ax2.plot(train_accuracy, label="Training accuracy")
    ax2.plot(test_accuracy, label="Validation accuracy")
    ax2.set_title("Accuracy Progression", color=text_color)
    ax2.set_xlabel("Epoch number", color=text_color)
    ax2.set_ylabel("Accuracy", color=text_color)
    ax2.legend(facecolor=face_color)
    ax2.tick_params(axis='both', colors=text_color)
    
    # Show the plot
    plt.tight_layout()

    if graph_path:
        plt.savefig(f"{graph_path}/result.png", dpi=600, bbox_inches='tight')

    plt.show()
