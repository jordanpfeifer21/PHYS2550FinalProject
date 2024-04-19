import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, latent_dims, in_feat, out_feat):
        self.linear1 = nn.Linear(in_feat, out_feat)
        self.linear2 = nn.Linear(out_feat, latent_dims)
        self.linear3 = nn.Linear(out_feat, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dims, in_feat, out_feat):
        self.linear1 = nn.Linear(latent_dims, out_feat)
        self.linear2 = nn.Linear(out_feat, in_feat)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    

class VariationalAutoEncoder(): 
    def __init__(self, latent_dims, in_feat, out_feat):
        self.encoder = Encoder(latent_dims, in_feat, out_feat)
        self.decoder = Decoder(latent_dims, in_feat, out_feat)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def train(autoencoder, data, epochs=20):
        opt = torch.optim.Adam(autoencoder.parameters())
        for epoch in range(epochs):
            for x, y in data:
                x = x.to(device) 
                opt.zero_grad()
                x_hat = autoencoder(x)
                loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
                loss.backward()
                opt.step()
        return autoencoder
        
    def plot_latent(autoencoder, data, num_batches=100):
        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break

    def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
        w = 28
        img = np.zeros((n*w, n*w))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(device)
                x_hat = autoencoder.decoder(z)
                x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
        plt.imshow(img, extent=[*r0, *r1])