import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
plt.style.use("seaborn-v0_8")

class AutoencoderAnomalyDetection(nn.Module):
    # Deep autoencoder.
    def __init__(self, shape=[2100], latent_dim=128):
        super().__init__()
        self.shape = shape
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(shape), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(shape)),
        )

    def forward(self, x):
        # print("Input shape:", x.shape)
        encoded = self.encoder(x)
        # print("Encoded shape:", encoded.shape)
        decoded = self.decoder(encoded)
        # print("Decoded shape:", decoded.shape)
        return decoded


class TransformerAnomalyDetection(nn.Module):
    # Autoencoder with attention.
    def __init__(self, input_size=(3, 700), latent_dim=2100, num_heads=4, num_layers=6):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=np.prod(input_size), nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=np.prod(input_size), nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        # Linear layers for encoding and decoding
        self.encoder_linear = nn.Linear(np.prod(input_size), latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, np.prod(input_size))

    def forward(self, x):
        # Flatten input
        x_flat = x.view(x.size(0), -1)  # Reshape to (batch_size, input_size)       
        # Encode
        encoded = self.encoder_linear(x_flat)  # (batch_size, latent_dim)
        encoded = encoded.unsqueeze(1)  # Add sequence length dimension (batch_size, seq_len=1, latent_dim)        
        # Transformer Encoder
        encoded = self.transformer_encoder(encoded)  # (batch_size, seq_len=1, latent_dim)
        # Decode
        decoded = self.decoder_linear(encoded.squeeze(1))  # (batch_size, prod(input_size))      
        return decoded


def train_loop(dataloader, model, loss_func, optimizer, batch_size):
    # Train a model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    for batch, X in enumerate(dataloader): # for batch, (X,y) in ... for labelled data.
        X = X.to(device)
        pred = model(X)
        loss = loss_func(pred, X) # X -> y
        
        # Backpropogation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 100 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        
    train_loss /= num_batches
    print(f"Average train loss over the number of batches: {train_loss:>8f} \n")
        
    return train_loss

    
def test_loop(dataloader, model, loss_func):
    # Test a model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    with torch.no_grad(): # No gradient calculated.
        for X in dataloader: # for (X,y) in ... for labelled data.
            X = X.to(device)
            pred = model(X)
            test_loss += loss_func(pred, X).item() # X -> y
            # correct += (pred.argmax(1) == X).type(torch.float).sum().item() # Does not make sense with autoencoders.

    test_loss /= num_batches
    print(f"Average test loss over the number of batches: {test_loss:>8f} \n")
    
    return test_loss

def train_model(dataloader_train, dataloader_test, model, loss_func, optimizer, epochs, 
                scheduler=None, batch_size=32, graph_path=None):
    # Training and testing (or validation).
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    
    for t in np.arange(0, epochs, step=1):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t] = train_loop(dataloader_train, model, loss_func, optimizer, batch_size=batch_size)
        test_loss[t] = test_loop(dataloader_test, model, loss_func)
        print("Done!")

        if scheduler:
            scheduler.step() # Update learning rate.
            current_lr = scheduler.get_last_lr() # Print the current learning rate
            print(f"Current learning rate: {current_lr[0]}")

    visualization(train_loss, test_loss, graph_path) # Plot and save graphs.

    return train_loss, test_loss

def evaluate_model(dataloader_test, model, loss_func, batch_size=32):
    # Model evaluation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = np.zeros(len(dataloader_test))
    with torch.no_grad():
        for ind, X in enumerate(dataloader_test):
            X = X.to(device)
            pred = model(X)
            test_loss[ind] = loss_func(pred, X).item()
    return test_loss
            
def visualization(train_loss, test_loss, graph_path):
    # Plot two types of graphs - with background (for report) and without (for slides).
    graph_text_face = [('white', 'none'), ('black', 'white')]
    for (text_color, face_color) in graph_text_face:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=face_color)
        ax.plot(train_loss, label="Training loss")
        ax.plot(test_loss, label="Validation loss")
        ax.set_title("Loss Progression", color=text_color)
        ax.set_xlabel("Epoch number", color=text_color)
        ax.set_ylabel("Reconstruction loss", color=text_color)
        ax.legend(facecolor=face_color)
        ax.tick_params(axis='both', colors=text_color)
        plt.tight_layout() # Remove empty edges.
        if graph_path:
            plt.savefig(graph_path.replace('.png', f'_{face_color}.png'), dpi=600, bbox_inches='tight')
        plt.show()

def exp_scheduler(optimizer, gamma):
    # Scheduled learning rate: lr_0 * exp(-gamma * iter_num).
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)